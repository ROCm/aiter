import os
import sys
from dataclasses import dataclass
import copy
from pathlib import Path
import pandas as pd
import argparse
import shutil
from itertools import product

from batched_gemm_bf16_common import kernelInstance


class autogen_instances:

    def __init__(self):
        self.mn_tile = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256]
        self.k_tile = [32, 64, 128]
        self.block_size = [64, 128, 256]
        self.mn_warp = [1, 2, 4]
        #self.pipeline = [1, 2, 3, 4, 5]
        self.pipeline = [1, 2, 3]
        self.scheduler = ["Intrawave", "Interwave"]
        self.nbyte_a = 2
        self.nbyte_b = 2
        self.nbyte_acc = 4
        self.nbyte_c = 2
        self.kpack = 128 // (self.nbyte_a * 8)

    def is_valid(self, blk, tile_m, tile_n, tile_k, m_warp, n_warp, pipeline, scheduler):

        num_warps = blk // 64
        if (m_warp * n_warp != num_warps):
            return False

        # alignment with mfma
        if (tile_m % (m_warp * 16) != 0 or tile_n % (n_warp * 16) != 0):
            return False

        if (tile_k % self.kpack != 0):
            return False

        # make sure full loads
        k0 = tile_k // self.kpack
        if ((tile_m * k0) % blk != 0 or (tile_n * k0) % blk != 0):
            return False

        # get LDS usage for a/b tile, pipeline v4 has double buffer
        lds_a = tile_m * tile_k * self.nbyte_a * (2 if pipeline == 4 else 1)
        lds_b = tile_n * tile_k * self.nbyte_b * (2 if pipeline == 4 else 1)
        lds_c = tile_m * tile_n * self.nbyte_c

        # lds size must no more than 64KB
        if ((lds_a + lds_b) > 64 * 1024):
            return False

        reg_a = tile_m * tile_k * self.nbyte_a // blk
        reg_b = tile_n * tile_k * self.nbyte_b // blk
        reg_c = tile_m * tile_n * self.nbyte_acc // blk

        # register usage for a/b tile buffer must be no more than 256 * 4 bytes
        if ((reg_a + reg_b) > 256 * 4):
            return False

        # register usage for a/b/c tile buffer must be no more than 512 * 4 bytes
        if ((reg_a + reg_b + reg_c) > 512 * 4):
            return False

        #calculate occupancy based on LDS and register usage
        occupancy = 64 * 1024 // (lds_a + lds_b)
        occupancy = min(occupancy, 256 * 4 // (reg_a + reg_b + reg_c))

        #Interwave requires occupancy >= 2
        if (occupancy < 2 and scheduler == "Interwave"):
            return False

        if (pipeline > 2 and scheduler == "Interwave"):
            return False

        return True

    def is_good(self, blk, tile_m, tile_n, tile_k, m_warp, n_warp, pipeline, scheduler):

        #if(not (blk == 256 and tile_m == 32 and tile_n == 128 and tile_k == 256)):
            #return False

        m_per_warp = tile_m // m_warp
        n_per_warp = tile_n // n_warp

        #limit warp workloads
        if ((m_per_warp > 64 or n_per_warp > 64) and blk < 256):
            return False

        if ((tile_m < 128 or tile_n < 128) and pipeline > 3):
            return False

        if ((tile_m < 128 and tile_n < 128) and tile_k < 64):
            return False

        #if ((tile_m < 32 or tile_n < 32) and pipeline > 2):
            #return False

        if ((tile_m >= 64 and tile_n >= 64) and pipeline < 3):
            return False

        return True

    def get_mfma(self, blk, tile_m, tile_n, m_warp, n_warp):
        m_per_warp = tile_m // m_warp
        n_per_warp = tile_n // n_warp

        # use 32x32 mfma if possible
        if (m_per_warp % 32 == 0 and n_per_warp % 32 == 0):
            mfma = 32
        else:
            mfma = 16

        m_repeat = m_per_warp // mfma
        n_repeat = n_per_warp // mfma

        return [mfma, mfma, m_repeat, n_repeat]

    def is_valid_c_transfer(self, blk, tid_m, tid_n, m_warp, n_warp, c_m_repeat,
                            c_n_repeat, mfma_cfg, n_vec):
        mfma, _, m_repeat, n_repeat = mfma_cfg

        c_shuffle_m = m_warp * mfma * c_m_repeat
        c_shuffle_n = n_warp * mfma * c_n_repeat

        if (tid_m * tid_n != blk):
            return False

        if (c_shuffle_m % tid_m != 0):
            return False

        if (c_shuffle_n % (tid_n * n_vec) != 0):
            return False

        if (m_repeat % c_m_repeat != 0):
            return False

        if (n_repeat % c_n_repeat != 0):
            return False

        lds_c_shuffle = c_shuffle_m * c_shuffle_n * self.nbyte_acc

        if (lds_c_shuffle > 32 * 1024):
            return False

        return True

    def try_c_transfer(self, blk, m_warp, n_warp, mfma_cfg, c_m_repeat,
                       c_n_repeat, n_vec):
        mfma, _, m_repeat, n_repeat = mfma_cfg

        c_shuffle_m = m_warp * mfma * c_m_repeat
        c_shuffle_n = n_warp * mfma * c_n_repeat

        # load tile_n dim first
        tid_n = c_shuffle_n // n_vec
        tid_m = blk // tid_n

        if (self.is_valid_c_transfer(blk, tid_m, tid_n, m_warp, n_warp,
                                     c_m_repeat, c_n_repeat, mfma_cfg, n_vec)):
            return tid_m, tid_n
        else:
            return 0, 0

    def get_c_transfer(self, blk, tile_m, tile_n, m_warp, n_warp, mfma_cfg):
        mfma, _, m_repeat, n_repeat = mfma_cfg

        n_vec = 128 // (8 * self.nbyte_c)

        c_m_repeat = 1
        c_n_repeat = 1

        c_shuffle_n = n_warp * mfma * c_n_repeat

        ctgs_store_size = c_shuffle_n * self.nbyte_c

        tid_m, tid_n = self.try_c_transfer(blk, m_warp, n_warp, mfma_cfg,
                                           c_m_repeat, c_n_repeat, n_vec)

        if(ctgs_store_size >= 128 and (tid_m * tid_n == blk)):
            return [[c_m_repeat, c_n_repeat], [1, tid_m, 1, tid_n],
                    [n_vec, n_vec, 1]]


        #if possible, enlarge c_n_repeat to fit 128B cacheline
        if (n_repeat % 2 == 0):
            c_n_repeat = 2
            tid_m, tid_n = self.try_c_transfer(blk, m_warp, n_warp, mfma_cfg,
                                               c_m_repeat, c_n_repeat, n_vec)
            if (tid_n * tid_m == blk):
                return [[c_m_repeat, c_n_repeat], [1, tid_m, 1, tid_n],
                        [n_vec, n_vec, 1]]

        #try enlarge c_m_repeat
        if (m_repeat % 2 == 0):
            c_m_repeat = 2
            tid_m, tid_n = self.try_c_transfer(blk, m_warp, n_warp, mfma_cfg,
                                               c_m_repeat, c_n_repeat, n_vec)
            if (tid_n * tid_m == blk):
                return [[c_m_repeat, c_n_repeat], [1, tid_m, 1, tid_n],
                        [n_vec, n_vec, 1]]

        #try reduce vec_len
        n_vec = n_vec // 2
        tid_m, tid_n = self.try_c_transfer(blk, m_warp, n_warp, mfma_cfg,
                                           c_m_repeat, c_n_repeat, n_vec)
        if (tid_n * tid_m == blk):
            return [[c_m_repeat, c_n_repeat], [1, tid_m, 1, tid_n],
                    [n_vec, n_vec, 1]]

        #failed, raise an Exception
        if (tid_n * tid_m != blk):
            raise Exception("cannot find proper cshuffle")

        return [-1]

    def get_ab_transfer(self, blk, mn, tile_k):

        # load tile_k dim first
        k0 = tile_k // self.kpack
        tid_k0 = k0
        tid_mn = blk // tid_k0

        return [tid_k0, tid_mn, 1]

    def gen_list(self):
        num_i = 0

        instance = {}

        for blk, tile_m, tile_n, tile_k, m_warp, n_warp, pipeline, scheduler in product(self.block_size, self.mn_tile, self.mn_tile, self.k_tile, self.mn_warp, self.mn_warp, self.pipeline, self.scheduler):
            if (self.is_valid(blk, tile_m, tile_n, tile_k, m_warp, n_warp, pipeline, scheduler) and self.is_good(blk, tile_m, tile_n, tile_k, m_warp, n_warp, pipeline, scheduler)):
                try:
                    mfma_cfg = self.get_mfma(blk, tile_m, tile_n, m_warp, n_warp)
                    a_load = self.get_ab_transfer(blk, tile_m, tile_k)
                    b_load = self.get_ab_transfer(blk, tile_n, tile_k)
                    c_shuffle = self.get_c_transfer(blk, tile_m, tile_n, m_warp, n_warp, mfma_cfg)
                    print(f"{num_i:>4}: kernelInstance({blk:>4},\t{tile_m:>4},\t{tile_n:>4},\t{tile_k:>4},\t{mfma_cfg[0]:>4},\t{mfma_cfg[1]:>4},\t{mfma_cfg[2]:>2},\t{mfma_cfg[3]:>2},\t{a_load},\t{b_load},\t{c_shuffle[1]},\t{c_shuffle[2]},\t{c_shuffle[0][0]},\t{c_shuffle[0][1]},\t\"{scheduler}\",\t{pipeline}),")
                    instance[num_i] = kernelInstance(
                            blk, tile_m, tile_n, tile_k,
                            mfma_cfg[0],
                            mfma_cfg[1],
                            mfma_cfg[2],
                            mfma_cfg[3],
                            a_load,
                            b_load,
                            c_shuffle[1],
                            c_shuffle[2],
                            c_shuffle[0][0],
                            c_shuffle[0][1],
                            scheduler,
                            pipeline
                            )
                    num_i += 1
                except Exception as e:
                    print(f"cannot generate proper instance with {blk, tile_m, tile_n, tile_k, m_warp, n_warp, mfma_cfg}, e = {e}")
        print(f"[AutoGen] {num_i} instances created")
        return instance


if __name__ == "__main__":
    gen_instance_list = autogen_instances()
    instances = gen_instance_list.gen_list()
    #for i in instances:
        #print(f"{instances[i]}")
