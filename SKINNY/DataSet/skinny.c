#include <stdint.h>
#include "skinny.h"

/* SKINNY Sbox */
uint8_t S8 [16][16] ={
0x65 ,0x4c ,0x6a ,0x42 ,0x4b ,0x63 ,0x43 ,0x6b ,0x55 ,0x75 ,0x5a ,0x7a ,0x53 ,0x73 ,0x5b ,0x7b ,
0x35 ,0x8c ,0x3a ,0x81 ,0x89 ,0x33 ,0x80 ,0x3b ,0x95 ,0x25 ,0x98 ,0x2a ,0x90 ,0x23 ,0x99 ,0x2b ,
0xe5 ,0xcc ,0xe8 ,0xc1 ,0xc9 ,0xe0 ,0xc0 ,0xe9 ,0xd5 ,0xf5 ,0xd8 ,0xf8 ,0xd0 ,0xf0 ,0xd9 ,0xf9 ,
0xa5 ,0x1c ,0xa8 ,0x12 ,0x1b ,0xa0 ,0x13 ,0xa9 ,0x05 ,0xb5 ,0x0a ,0xb8 ,0x03 ,0xb0 ,0x0b ,0xb9 ,
0x32 ,0x88 ,0x3c ,0x85 ,0x8d ,0x34 ,0x84 ,0x3d ,0x91 ,0x22 ,0x9c ,0x2c ,0x94 ,0x24 ,0x9d ,0x2d ,
0x62 ,0x4a ,0x6c ,0x45 ,0x4d ,0x64 ,0x44 ,0x6d ,0x52 ,0x72 ,0x5c ,0x7c ,0x54 ,0x74 ,0x5d ,0x7d ,
0xa1 ,0x1a ,0xac ,0x15 ,0x1d ,0xa4 ,0x14 ,0xad ,0x02 ,0xb1 ,0x0c ,0xbc ,0x04 ,0xb4 ,0x0d ,0xbd ,
0xe1 ,0xc8 ,0xec ,0xc5 ,0xcd ,0xe4 ,0xc4 ,0xed ,0xd1 ,0xf1 ,0xdc ,0xfc ,0xd4 ,0xf4 ,0xdd ,0xfd ,
0x36 ,0x8e ,0x38 ,0x82 ,0x8b ,0x30 ,0x83 ,0x39 ,0x96 ,0x26 ,0x9a ,0x28 ,0x93 ,0x20 ,0x9b ,0x29 ,
0x66 ,0x4e ,0x68 ,0x41 ,0x49 ,0x60 ,0x40 ,0x69 ,0x56 ,0x76 ,0x58 ,0x78 ,0x50 ,0x70 ,0x59 ,0x79 ,
0xa6 ,0x1e ,0xaa ,0x11 ,0x19 ,0xa3 ,0x10 ,0xab ,0x06 ,0xb6 ,0x08 ,0xba ,0x00 ,0xb3 ,0x09 ,0xbb ,
0xe6 ,0xce ,0xea ,0xc2 ,0xcb ,0xe3 ,0xc3 ,0xeb ,0xd6 ,0xf6 ,0xda ,0xfa ,0xd3 ,0xf3 ,0xdb ,0xfb ,
0x31 ,0x8a ,0x3e ,0x86 ,0x8f ,0x37 ,0x87 ,0x3f ,0x92 ,0x21 ,0x9e ,0x2e ,0x97 ,0x27 ,0x9f ,0x2f ,
0x61 ,0x48 ,0x6e ,0x46 ,0x4f ,0x67 ,0x47 ,0x6f ,0x51 ,0x71 ,0x5e ,0x7e ,0x57 ,0x77 ,0x5f ,0x7f ,
0xa2 ,0x18 ,0xae ,0x16 ,0x1f ,0xa7 ,0x17 ,0xaf ,0x01 ,0xb2 ,0x0e ,0xbe ,0x07 ,0xb7 ,0x0f ,0xbf ,
0xe2 ,0xca ,0xee ,0xc6 ,0xcf ,0xe7 ,0xc7 ,0xef ,0xd2 ,0xf2 ,0xde ,0xfe ,0xd7 ,0xf7 ,0xdf ,0xff
};


uint8_t RC [62] = {
    0x01,0x03,0x07,0x0f,0x1f,0x3e,0x3d,0x3B,0x37,0x2F,0x1E,0x3C,0x39,0x33,0x27,0x0e,
    0x1d,0x3a,0x35,0x2B,0x16,0x2C,0x18,0x30,0x21,0x02,0x05,0x0B,0x17,0x2E,0x1C,0x38,
    0x31,0x23,0x06,0x0D,0x1B,0x36,0x2D,0x1A,0x34,0x29,0x12,0x24,0x08,0x11,0x22,0x04,
        0x09,0x13,0x26,0x0C,0x19,0x32,0x25,0x0A,15,0x2A,0x14,0x28,0x10,0x20
};

uint8_t mixColumnsMatrix [4][4] = {
    1, 0, 1, 1,
    1, 0, 0, 0,
    0, 1, 1, 0,
    1, 0, 1, 0
};

uint8_t tkPermutation [16] = { 
    9, 15,  8, 13,
    10, 14, 12, 11,
    0,  1,  2,  3,
    4,  5,  6,  7,
};


void printArrayState(unsigned char array[]) {

    int i = 0;
    printf("Array State:");
    for(i = 0; i < 16; i++) {
        printf("%x", array[i]);
    }
    printf("\n");
}

void subCells(unsigned char *internalState) {

    int i; 

    for( i = 0; i < 16; i++ ){
        // Substitute the current bytes with bytes from the S-Box
        // S8[y][x]
        // Where y-value = first nibble of the byte
        // x-value = the second nibble of the byte
        internalState[i] = S8[(internalState[i] & 0xf0) >> 4][internalState[i] & 0x0f];
    }

}


void addConstants(unsigned char *internalState, int r) {

    unsigned char  rc = RC[r];
    //                                              bit 3                     bit2                     bit 1                    bit 0
    internalState[0] = internalState[0] ^ (0x00|((rc >> 3)  & 0x01) << 3|((rc >> 2)  & 0x01) << 2|((rc >> 1)  & 0x01) << 1|((rc >> 0)  & 0x01) << 0);
    //                                              bit 5                     bit 4
    internalState[4] = internalState[4] ^ (0x00|((rc >> 5)  & 0x01) << 1|((rc >> 4)  & 0x01) << 0);

    internalState[8] = internalState[8] ^ 0x2;

}

void tkLSFR(unsigned char tweakey[]) {

    int j;
    int z;

    // TK2 first 2 rows
    for( j = 16; j < 24; j++) {
        unsigned char bit0 = ((tweakey[j] >> 0)  & 0x01);
        unsigned char bit1 = ((tweakey[j] >> 1)  & 0x01);
        unsigned char bit2 = ((tweakey[j] >> 2)  & 0x01);
        unsigned char bit3 = ((tweakey[j] >> 3)  & 0x01);
        unsigned char bit4 = ((tweakey[j] >> 4)  & 0x01);
        unsigned char bit5 = ((tweakey[j] >> 5)  & 0x01);
        unsigned char bit6 = ((tweakey[j] >> 6)  & 0x01);
        unsigned char bit7 = ((tweakey[j] >> 7)  & 0x01);

        tweakey[j] = (0x00|bit6 << 7| bit5 << 6|bit4 << 5|bit3 << 4|bit2 << 3|bit1 << 2|bit0 << 1|(bit7 ^ bit5) << 0);
    }
    // TK3 first two rows
    for( z = 32; z < 40; z++) {
        unsigned char bit0 = ((tweakey[z] >> 0)  & 0x01);
        unsigned char bit1 = ((tweakey[z] >> 1)  & 0x01);
        unsigned char bit2 = ((tweakey[z] >> 2)  & 0x01);
        unsigned char bit3 = ((tweakey[z] >> 3)  & 0x01);
        unsigned char bit4 = ((tweakey[z] >> 4)  & 0x01);
        unsigned char bit5 = ((tweakey[z] >> 5)  & 0x01);
        unsigned char bit6 = ((tweakey[z] >> 6)  & 0x01);
        unsigned char bit7 = ((tweakey[z] >> 7)  & 0x01);

        tweakey[z] = (0x00|(bit0 ^ bit6) << 7| bit7 << 6|bit6 << 5|bit5 << 4|bit4 << 3|bit3 << 2|bit2 << 1|bit1 << 0);
    }

}
void updateTweakey(unsigned char *tweakey) {

    int i;
    int j;
    int z;

    unsigned char temp[48];
    memcpy(temp, tweakey, 48);

    // TK1
    for ( i = 0; i < 16; i++ ) {
        tweakey[i] = temp[tkPermutation[i]];
    }
    // TK2
    for( j = 16; j < 32; j++) {
        tweakey[j] = temp[tkPermutation[j-16] + 16];
    }
    // TK3
    for( z = 32; z < 48; z++) {
        tweakey[z] = temp[tkPermutation[z-32] + 32];
    }

    tkLSFR(tweakey);
}


void addRoundTweakey(unsigned char *internalState, unsigned char *tweakey) {

    int i;
    int j;
    int z;

    // TK1 first 2 rows
    for( i = 0; i < 8; i++) {
        internalState[i] = internalState[i] ^ tweakey[i];
    }
    // TK2 first 2 rows
    for( j = 16; j < 24; j++) {
        internalState[j-16] = internalState[j-16] ^ tweakey[j];
    }
    // TK3 first 2 rows
    for( z = 32; z < 40; z++) {
        internalState[z-32] = internalState[z-32] ^ tweakey[z];
    }

    updateTweakey(tweakey);

}

void shiftRows(unsigned char *internalState) {

    // cast the original 1*16 internal state to 4*4 matrix for easier understanding
    typedef unsigned char fourByFour_t[4][4];
    fourByFour_t *fourByFour;
    fourByFour = (fourByFour_t *) internalState;

    int i;

    for ( i = 0; i < 4; i++ ) {
        // row shifting using custom modulo function
        unsigned char rowZero  = (*fourByFour)[i][modulo((0 - i), 4)];
        unsigned char rowOne   = (*fourByFour)[i][modulo((1 - i), 4)];
        unsigned char rowTwo   = (*fourByFour)[i][modulo((2 - i), 4)];
        unsigned char rowThree = (*fourByFour)[i][modulo((3 - i), 4)];

        (*fourByFour)[i][0] = rowZero;
        (*fourByFour)[i][1] = rowOne;  
        (*fourByFour)[i][2] = rowTwo;
        (*fourByFour)[i][3] = rowThree;
    }

}


int modulo(int x, int mod) {
    if ( x < 0) {
        return mod + x;
    }
    else if (x > mod) {
        return mod % x;
    }
    return x;
}


void mixColumns(unsigned char *internalState) {
    
    // cast the original 1*16 internal state to 4*4 matrix for easier understanding
    typedef unsigned char fourByFour_t[4][4];
    fourByFour_t *fourByFour;
    fourByFour = (fourByFour_t *) internalState;

    int i;

    // Loop through all columns
    for ( i = 0; i < 4; i++ ) {

        // matrix multiplication by column
        unsigned char first = ((*fourByFour)[1][i]) ;
        unsigned char second = (((*fourByFour)[1][i]) ^ ((*fourByFour)[2][i] ) ^ ((*fourByFour)[3][i] ));
        unsigned char third = (((*fourByFour)[1][i]) ^ ((*fourByFour)[3][i] ));
        unsigned char fourth = (((*fourByFour)[0][i]) ^ ((*fourByFour)[3][i] ));
    
        (*fourByFour)[0][i] = first;
        (*fourByFour)[1][i] = second;
        (*fourByFour)[2][i] = third;
        (*fourByFour)[3][i] = fourth;
    }

}
/**
 * SKINNY-128-384 block cipher encryption.
 * Under 48-byte tweakey at k, encrypt 16-byte plaintext at p and store the 16-byte output at c.
 */
void skinny(unsigned char *c, const unsigned char *p, const unsigned char *k) {

    unsigned char internalState[16];
    memcpy(internalState, p, 16);

    unsigned char tweakey[48];
    memcpy(tweakey, k, 48);

    int round;
    
    for(round = 0; round < 56; round++) {
        subCells(internalState);
        addConstants(internalState, round);
        addRoundTweakey(internalState, tweakey);
        shiftRows(internalState);
        mixColumns(internalState);
    }

    printArrayState(internalState);
    memcpy(c, internalState, 16);
}
void Generate_plaintext( unsigned char *plaintext) {
	for (int i = 0; i < 16; i++) {
		plaintext[i]= rand() % 255;
	}
}