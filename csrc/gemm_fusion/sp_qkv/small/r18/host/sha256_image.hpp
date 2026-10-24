// SPDX-License-Identifier: MIT
// Copied byte-exact (algorithm/lines) from PR695-M4 host/driver.cpp lines 70-130 (Sp4cSmokeSha256, FIPS 180-4) and 154-171
// (file hash + exact-image match); only renamed sm_ prefix for the two helper functions. No OpenSSL.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstddef>
class Sp4cSmokeSha256 {
public:
    Sp4cSmokeSha256() { reset(); }
    void reset() {
        h_[0]=0x6a09e667u; h_[1]=0xbb67ae85u; h_[2]=0x3c6ef372u; h_[3]=0xa54ff53au;
        h_[4]=0x510e527fu; h_[5]=0x9b05688cu; h_[6]=0x1f83d9abu; h_[7]=0x5be0cd19u;
        total_=0; buffered_=0;
    }
    void update(const uint8_t* data, size_t len) {
        total_ += len;
        while (len > 0) {
            size_t take = 64 - buffered_;
            if (take > len) take = len;
            std::memcpy(buffer_ + buffered_, data, take);
            buffered_ += take; data += take; len -= take;
            if (buffered_ == 64) { compress(buffer_); buffered_ = 0; }
        }
    }
    void finish(uint8_t out[32]) {
        const uint64_t bitlen = total_ * 8u;
        buffer_[buffered_++] = 0x80u;
        if (buffered_ > 56) { while (buffered_ < 64) buffer_[buffered_++] = 0u; compress(buffer_); buffered_ = 0; }
        while (buffered_ < 56) buffer_[buffered_++] = 0u;
        for (int i = 7; i >= 0; --i) buffer_[buffered_++] = static_cast<uint8_t>(bitlen >> (i * 8));
        compress(buffer_);
        for (int i = 0; i < 8; ++i) {
            out[i*4+0]=static_cast<uint8_t>(h_[i]>>24); out[i*4+1]=static_cast<uint8_t>(h_[i]>>16);
            out[i*4+2]=static_cast<uint8_t>(h_[i]>>8);  out[i*4+3]=static_cast<uint8_t>(h_[i]);
        }
    }
private:
    static uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
    void compress(const uint8_t* block) {
        static const uint32_t K[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (static_cast<uint32_t>(block[i*4])<<24)|(static_cast<uint32_t>(block[i*4+1])<<16)
                 | (static_cast<uint32_t>(block[i*4+2])<<8)|static_cast<uint32_t>(block[i*4+3]);
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
            const uint32_t s1 = rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h_[0],b=h_[1],c=h_[2],d=h_[3],e=h_[4],f=h_[5],g=h_[6],h=h_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1=rotr(e,6)^rotr(e,11)^rotr(e,25), ch=(e&f)^(~e&g), t1=h+S1+ch+K[i]+w[i];
            const uint32_t S0=rotr(a,2)^rotr(a,13)^rotr(a,22), maj=(a&b)^(a&c)^(b&c), t2=S0+maj;
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h_[0]+=a; h_[1]+=b; h_[2]+=c; h_[3]+=d; h_[4]+=e; h_[5]+=f; h_[6]+=g; h_[7]+=h;
    }
    uint32_t h_[8]; uint64_t total_; size_t buffered_; uint8_t buffer_[64];
};

static inline bool sm_file_sha256_hex(const char* path, char out[65]) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    Sp4cSmokeSha256 sha; uint8_t buf[65536]; size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) sha.update(buf, n);
    const bool ok = !std::ferror(f);
    std::fclose(f);
    if (!ok) return false;
    uint8_t d[32]; sha.finish(d);
    static const char* hx = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) { out[2*i] = hx[d[i] >> 4]; out[2*i+1] = hx[d[i] & 15]; }
    out[64] = 0;
    return true;
}
static inline bool sm_image_matches(const char* path, const char* expected_hex) {
    char got[65];
    return sm_file_sha256_hex(path, got) && std::strcmp(got, expected_hex) == 0;
}
