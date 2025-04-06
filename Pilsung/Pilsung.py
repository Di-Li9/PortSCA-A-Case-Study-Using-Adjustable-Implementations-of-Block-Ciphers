import numpy as np
Tree_Integer8=np.array([ 0x0F, 0x87, 0x47, 0x27, 0x17, 0xC3, 0x63, 0x33, 0x1B, 0xA3, 
  0x53, 0x2B, 0x93, 0x4B, 0x8B, 0xE1, 0x71, 0x39, 0x1D, 0xB1, 
  0x59, 0x2D, 0x99, 0x4D, 0x8D, 0xD1, 0x69, 0x35, 0xA9, 0x55, 
  0x95, 0xC9, 0x65, 0xA5, 0xC5, 0xF0, 0x78, 0x3C, 0x1E, 0xB8, 
  0x5C, 0x2E, 0x9C, 0x4E, 0x8E, 0xD8, 0x6C, 0x36, 0xAC, 0x56, 
  0x96, 0xCC, 0x66, 0xA6, 0xC6, 0xE8, 0x74, 0x3A, 0xB4, 0x5A, 
  0x9A, 0xD4, 0x6A, 0xAA, 0xCA, 0xE4, 0x72, 0xB2, 0xD2, 0xE2])

Tree_Integer4=np.array([0x03, 0x09, 0x05, 0x0C, 0x06, 0x0A])
AES_sbox=np.array([ 0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
  0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
  0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
  0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
  0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
  0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
  0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
  0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
  0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
  0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
  0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
  0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
  0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
  0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
  0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16])

sboxes=np.zeros([4,4,256])
pboxes=np.zeros([16])
current_permutation_8=np.zeros([8])
sbox_xor_constant=3

def Get_One(s,p,n,buf):
    #print("s,p,n,buf",s.shape,p.shape,n.shape,buf.shape)
    a,b=0,0
    for i in range(n):
        if(int(s[i])):
            buf[int(n/2+a)]=p[i]
            a=a+1
        else:
    
            buf[int(b)]=p[i]
            b=b+1
    for i in range(n):
        p[i]=buf[i]

def Get_PeSb(input_mask,current_permutation_8):
    x=0
    for i in range(8):
        if((1<<i)&input_mask):
            x|=1<<int(current_permutation_8[i])
    return x^sbox_xor_constant

def Get_P8forSEnc(input_mask,current_permutation_8):
    coinflips=np.zeros([24])
     #Random 4-bit subset
    v0 = Tree_Integer8[input_mask % 70]
    for i in range(8):
        coinflips[i]=(v0 >> (7 - i)) & 1


    v1 = (Tree_Integer4[(input_mask & 0x0F) % 6] << 0) | (Tree_Integer4[(input_mask >> 4) % 6] << 4)
    for i in range(8):
        coinflips[i+8]=(v1 >> (7 - i)) & 1
    for i in range(0,8,2):
        if( input_mask & (3 << i) ):
            coinflips[16+i+0] = 1
            coinflips[16+i+1] = 0
        else:
            coinflips[16+i+0] = 0;
            coinflips[16+i+1] = 1;
    for i in range(0,8):
        current_permutation_8[i] = i

    scratch=np.zeros([32])
    for i in range(3):
        bins = 1 << i
        size = 1 << (3 - i)
        for j in range(bins):
             Get_One(coinflips[i * 8 + j * size:], current_permutation_8[j * size:], size, scratch)
             #print('after',current_permutation_8)

def Get_P16Enc( input,  output):
    coinflips=np.zeros([64])
    for i in range(4):
  
        v0=Tree_Integer4[(input[i] ^ input[i+4]) % 6]
        for j in range(4):
            coinflips[4 * i + j] = (v0 >> (3 - j)) & 1
  #coin flips for second level
    for i in range(8):
        if((input[i] >> i) & 1) :
            coinflips[16 + 2*i + 0] = 1;
            coinflips[16 + 2*i + 1] = 0;
        else:
            coinflips[16 + 2*i + 0] = 0;
            coinflips[16 + 2*i + 1] = 1;
 #coin flips for third level
    for i in range(4):
        v1 = Tree_Integer4[(input[i+8] ^ input[i+12]) % 6]
        for j in range(4):
            coinflips[32+4*i+j]= (v1 >> (3 - j)) & 1
# coin flips for fourth level
    for i in range(8):
         if((input[8+i] >> i) & 1):
             coinflips[48 + 2*i + 0] = 1
             coinflips[48 + 2*i + 1] = 0
         else:
            coinflips[48 + 2*i + 0] = 0
            coinflips[48 + 2*i + 1] = 1
# // Initialize permutation with identity
    for i in range(16):
        output[i]=i
#Iterative version of the Rao-Sandelius shuffle
    scratch=np.zeros([32])
    for i in range(4):
        bins = 1 << i
        size = 1 << (4 - i)
        for j in range(bins):
            Get_One(coinflips[i * 16 + j * size:], output[j * size:], size, scratch)

def Get_VSboxAll(e_key,current_permutation_8):
    rounds=1
    for i in range(4):
        for j in range(4):
            Get_P8forSEnc(e_key[j + 4 * i + 16 * rounds],current_permutation_8)
            for k in range(256):
                sboxes[i][j][k] = Get_PeSb(AES_sbox[k],current_permutation_8)

def Get_VSboxAll_sbox1(e_key,current_permutation_8):
    rounds=1
    for i in range(4):
        for j in range(4):
            Get_P8forSEnc(e_key[j + 4 * i + 16 * rounds],current_permutation_8)
            for k in range(256):
                sboxes[i][j][k] = Get_PeSb(AES_sbox[k],current_permutation_8)

def Get_VPboxAll(e_key,pboxes):
    for rounds in range(5):
        Get_P16Enc(e_key[16:],pboxes)

def gen_enc_perm(e_key,current_permutation_8,pboxes):
    Get_VSboxAll(e_key,current_permutation_8)
    Get_VPboxAll(e_key,pboxes)

def gen_enc_perm_sbox1(e_key,current_permutation_8,pboxes):
    Get_VSboxAll_sbox1(e_key,current_permutation_8)
    Get_VPboxAll(e_key,pboxes)

def AddRoundKey(state,key):
    for i in range(4):
        for j in range(4):
            state[i][j]=int(state[i][j])^int(key[i*4+j])

def SubBytes(state):
    for i in range(4):
        for j in range(4):
            state[i][j]=sboxes[i][j][int(state[i][j])]

def ShiftRows(state,pboxes):
    copy=np.zeros([16])
    for i in range(4):
        for j in range(4):
            copy[i*4+j]=state[i][j]
    for i in range(4):
        for j in range(4):
            state[i][j]=copy[int(pboxes[i*4+j])]

# def pilsung_encrypt(input ,output,key):
#     sboxes=np.zeros([4,4,256])
#     pboxes=np.zeros([16])
#     current_permutation_8=np.zeros([8])
#     state=np.zeros([4,4])
#     for i in range(4):
#         for j in range(4):
#             state[i][j]=input[i*4+j]
#     AddRoundKey(state,key[0:16])
#     SubBytes(state)
#     for i in range(4):
#         for j in range(4):
#             output[i*4+j]=state[i][j]
#     #ShiftRows(state,pboxes)
#     return output

    



