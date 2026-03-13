#include"aes.h"
u8 sbox[256];

void Generate_plaintext(u8 *plaintext) {
	for (int i = 0; i < 16; i++) {
		plaintext[i]= rand() % 256;
	}
}

void printState(u8 in[16])
{
	int i;
	for (i = 0; i < 16; i++)
	{
		printf("%-3.2x", in[i]);
	}
}

void subBytes (u8 state[16]) 
{
  int i;
  for (i = 0; i < 16; i++)
	  state[i] = SboxMasked[state[i]];  
}


void shiftRows (u8 state[16]) 
{
  int i;
  u8 out[16];
  int shiftTab[16] = {0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11};
  for (i = 0; i < 16; i++) 
  {
    out[i] = state[shiftTab[i]];
  }
  for (i = 0; i < 16; i++) 
  {
    state[i]=out[i];
  }
  // memcpy(state, out, sizeof(out));
}

void addRoundKey (u8 state[16], u8 roundKey[16]) 
{
  int i;
  for (i = 0; i < 16; i++)
    state[i] ^= roundKey[i];
}


u8 gMul (u8 a, u8 b) 
{
  int i;
  u8 p = 0;
  u8 hi_bit_set;

  for (i = 0; i < 8; i++) {
    if ((b & 1) == 1)
      p ^= a;
    hi_bit_set = (a & 0x80);
    a <<= 1;
    if (hi_bit_set == 0x80)
      a ^= 0x1b;
    b >>= 1;
  }
return p;
}


void mixColumns (u8 state[16]) 
{
  int i;
  u8 out[16];
  for (i = 0; i < 4; i++) {
    out[4*i] = gMul(2, state[4*i]) ^ gMul(3, state[4*i + 1]) ^ gMul(1, state[4*i + 2]) ^ gMul(1, state[4*i + 3]);
    out[4*i + 1] =gMul(1, state[4*i]) ^ gMul(2, state[4*i + 1]) ^ gMul(3, state[4*i + 2]) ^ gMul(1, state[4*i + 3]);;
    out[4*i + 2] = gMul(1, state[4*i]) ^ gMul(1, state[4*i + 1]) ^ gMul(2, state[4*i + 2]) ^ gMul(3, state[4*i + 3]);
    out[4*i + 3] = gMul(3, state[4*i]) ^ gMul(1, state[4*i + 1]) ^ gMul(1, state[4*i + 2]) ^ gMul(2, state[4*i + 3]);
  }
  
  memcpy(state, out, sizeof(out));
}

void expandKey (u8 key[16], u8 expandedKey[176]) {
  u8 tmp[4];
  int i = 0;
  int k;

  for (i = 0; i < 4; i++) {
    expandedKey[4*i] = key[4*i];
    expandedKey[4*i + 1] = key[4*i + 1];
    expandedKey[4*i + 2] = key[4*i + 2];
    expandedKey[4*i + 3] = key[4*i + 3];
  }

  for (i = 4; i < 44; i++) {
    tmp[0] = expandedKey[4*(i-1)];
    tmp[1] = expandedKey[4*(i-1) + 1];
    tmp[2] = expandedKey[4*(i-1) + 2];
    tmp[3] = expandedKey[4*(i-1) + 3];

    if (i % 4 == 0) 
    {
      k = tmp[0];
      tmp[0] = SBox[tmp[1]] ^ rCon[i/4];
      tmp[1] = SBox[tmp[2]];
      tmp[2] = SBox[tmp[3]];
      tmp[3] = SBox[k];

    }
    expandedKey[4*i] = expandedKey[4*(i-4)] ^ tmp[0];
    expandedKey[4*i + 1] = expandedKey[4*(i-4) + 1] ^ tmp[1];
    expandedKey[4*i + 2] = expandedKey[4*(i-4) + 2] ^ tmp[2];
    expandedKey[4*i + 3] = expandedKey[4*(i-4) + 3] ^ tmp[3];
  }                                  
}


void remask(u8 state[16], u8 old_mask[16], u8 new_mask[16])
{
 
      for (int i = 0; i < 16; i++)
    {
        state[i]^=new_mask[i];
    }

      for (int i = 0; i < 16; i++)
    {
        state[i]^=old_mask[i];
    }
}



static void calcSboxMasked(uint8_t mask_in[16],uint8_t mask_out[16])
{
  for (int i = 0; i < 256; i++){
    // SboxMasked[i ^ mask_in[0]] = SM4_SBox[i] ^ mask_out[0];
    SboxMasked[i ^ mask_in[0]] = SBox[i] ^ mask_out[0];
    // SboxMasked[i ^ mask_in[0]] = Skinny_SBox[i] ^ mask_out[0];
  }
}


void InitMask(uint8_t mask[16])
{
    for (int i = 0; i < 16; i++) {
        uint8_t rand_val;
        do {
            rand_val = rand() % 256;  
        } while (rand_val == 0);    
        
        mask[i] = rand_val;  
    }
}

void Initmask_inout(uint8_t mask_in[16], uint8_t mask_out[16])
{
    uint8_t a;
    do {
        a = rand() % 256;
    } while (a == 0);     

    for (int i = 0; i < 16; i++) {
        mask_in[i] = a;
    }
    uint8_t b;
    do {
        b = rand() % 256;  
    } while (b == 0);     

    for (int i = 0; i < 16; i++) {
        mask_out[i] = b;
    }
}

void aes_128_encrypt (u8 input[16], u8 key[16],u8 mask[16],u8 zero[16],u8 mask_in[16],u8 mask_out[16],u8 output[16]) 
{
  u8 expandedKey[176];
  uint8_t expandedKeyMasked[AES_keyExpSize] = {0};
  int i=0;
  expandKey (key, expandedKey);
  calcSboxMasked(mask_in, mask_out);
  InitMaskingEncrypt(expandedKey, mask,mask_in,mask_out);
  remask(input,zero,mask);
  for (i =0; i<9; i++) 
  {
  addRoundKey(input, expandedKey + 16*i);
  remask(input,mask,mask_in);
  subBytes (input);
  remask(input,mask_out,mask);
  shiftRows(input);
  shiftRows(mask); 

  mixColumns(input);
  mixColumns(mask);

  }
  addRoundKey (input, expandedKey + 144);
  remask(input,mask,mask_in);
  subBytes (input);
  remask(input,mask_out,mask);
  shiftRows (input);
  shiftRows(mask);
  addRoundKey (input, expandedKey + 160);
  remask(input,mask,zero);
  for (i = 0; i < 16; i++){
    output[i] = input[i];}
}
