def pack_weight(w):
 n,k=w.shape
 return w.reshape(n//16,16,k//16,16).permute(0,2,1,3).contiguous().reshape(n,k)
