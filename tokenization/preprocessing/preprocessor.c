#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<errno.h>
#define debug() printf("Debug %d\n",__LINE__)

const int utf_8_gr_eng = 2 ;
const int utf_8_gr_max_first_byte = 207 ;
const int utf_8_gr_min_first_byte = 206 ;
const int utf_8_gr_206_min_second = 134 ;
const int utf_8_gr_206_max_second = 191 ;
const int utf_8_gr_207_min_second = 128 ;
const int utf_8_gr_207_max_second = 142 ;

int char_bytes(unsigned char input) {
    if( input < 128 ) {
        return 1 ;
    }
    if( input < 224 ) {
        return 2 ;
    }
    if( input < 240 ) {
        return 3 ;
    }
    return 4 ;
}

void fullstrip(FILE* dst, FILE* src, unsigned char character){
    if( char_bytes(character) > utf_8_gr_eng ) { //too big to be greek or english
        for( int i = 1 ; i < char_bytes(character) ; i++ ) {
            character = fgetc(src) ;
        }
        return;
    }
    if ( char_bytes(character) == 1) {  //english
        if( character == 10 || character == 13 || character == 20 || character == 32 ){ //spaces and newline
            fprintf(dst,"%c", character) ;
        }
        else if ( character < 97 && character > 122  ){ // lower case letters
            fprintf(dst,"%c", character) ;
        }
        else if ( character < 65 && character > 90 ){
            fprintf(dst,"%c", character) ;
        }
    }
    else {
        unsigned char nextbyte = fgetc(src) ;
        if( character > utf_8_gr_max_first_byte) { //too big to be greek
            return ;
        }
        if( character < utf_8_gr_min_first_byte ) { //too small to be greek
            return ;
        }
        if( character == 206 && (( nextbyte > utf_8_gr_206_max_second) || ( nextbyte < utf_8_gr_206_min_second) || ( nextbyte == 135 )) ) { //not a letter
            return ;
        }
        if( character == 207 && (( nextbyte > utf_8_gr_207_max_second) || ( nextbyte < utf_8_gr_207_min_second) )) { //not a letter
            return ;
        }
        fprintf(dst,"%c", character) ;
        fprintf(dst,"%c", nextbyte) ;
    }
}

unsigned int create_code(FILE* src, unsigned char character) {
    unsigned int temp ;
    if ( char_bytes(character) == 1) {
        temp = character ;
    }
    else {
        unsigned int nextbyte = fgetc(src) ;
        temp = character ;
        nextbyte = nextbyte << 8 ;
        temp = temp | nextbyte ;
    }
    return temp ;
}

// TO-DO expand for multiple layers
void decode_text(FILE* dst, FILE* src) {
    while( 1 ) {
        unsigned int temp ;
        if( feof(src) ) {
            return ;
        }
        fread(&temp,sizeof(unsigned int), 1 , src) ;
        if( temp < 128 ){
            unsigned char tchar = (char)temp ;
            fwrite(&tchar,sizeof(unsigned char),1,dst) ;
        }
        else {
            unsigned short schar = (short)temp ;
            fwrite(&schar,sizeof(unsigned short),1,dst) ;
        }
    }
}

void encode_text(FILE* dst, FILE* src, unsigned char character) {
    unsigned int output = create_code(src,character) ;
    fwrite(&output,sizeof(unsigned int),1,dst) ;
    if( output == 0xffff ) {
        return ;
    }
    return ;
}

int main(int argc, char** argv) {
    FILE* src = NULL ;
    FILE* dst = NULL ;
    int mode = 0 ;
    int srcprovided = 0 ;
    int dstprovided = 0 ;
    for( int i = 1 ; i < argc ; i++ ) {
        if( !memcmp(argv[i],"-ec",strlen("-ec")) ) {
            mode = 1 ;
        }
        else if( !memcmp(argv[i],"-dc",strlen("-dc")) ) {
            mode = 2 ;
        }
        else if( !memcmp(argv[i],"-strip",strlen("-strip")) ) {
            mode = 3 ;
        }
        else if( !memcmp(argv[i],"-s",strlen("-s")) ) {
            i++ ;
            src = fopen(argv[i],"r") ;
            if( src == NULL) {
                printf("Failed to find source file\n") ;
                printf("Error : %s \n",strerror(errno)) ;
                return -1 ;
            }
            srcprovided++ ;
        }
        else if( !memcmp(argv[i],"-d",strlen("-d")) ) {
            i++ ;
            dst = fopen(argv[i],"w") ;
            if( dst == NULL ) {
                printf("Failed to find or create destination file\n") ;
                printf("Error : %s \n",strerror(errno)) ;
                return -1 ;
            }
            dstprovided++ ;
        }
    }
    if( mode < 0 && mode > 4 ) {
        printf("Failed to set mode correctly \n") ;
    }
    if( !srcprovided ) {
        printf("Please provide source file \n") ;
    }
    if( !dstprovided ) {
        printf("Please provide destination file \n") ;
    }
    unsigned char character ;
    if( mode == 1 ) {
        while( 1 ) {
            if ( feof(src) ) {
                break;
            }
            character = fgetc(src) ;
            encode_text(dst,src,character) ;
        }
    }
    if( mode == 2 ) {
        while( 1 ) {
            if ( feof(src) ) {
                break;
            }
            character = fgetc(src) ;
            fullstrip(dst,src,character) ;
        }
    }
    if ( mode == 3 ) {
        decode_text(dst,src) ;
    }
    fclose(src) ;
    fclose(dst) ;
    return 0 ;
}