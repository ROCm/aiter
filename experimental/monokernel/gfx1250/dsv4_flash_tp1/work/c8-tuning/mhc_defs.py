import sys,json,torch
sys.path[:0]=['/work/c8-tuning','/work/batch-full','/work/modelwide','/work/full-attention-native/prefix']
from mhc_batch import mhc_batch
from mhc_precise import candidate_body
from post_batch import post_batch
from post_precise import post_body
from triton.experimental import gluon as g
@g.jit
def c1(R,W,S,B,AN,Mix,Temp,Raw,X,C,H,F):
 candidate_body(R,W,S,B,AN,Mix,Temp,Raw,X,C,H,FP32=F)
@g.jit
def post_c1(R,X,M,O,F):post_body(R,X,M,O,FP32=F)
