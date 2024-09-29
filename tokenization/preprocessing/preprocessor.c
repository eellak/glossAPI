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

void fullstrip(FILE* dst, FILE* src){
    while( 1 ) {
        if ( feof(src) ) {
            break;
        }
        unsigned char character = 0 ;
        fread(&character,sizeof(unsigned char),1,src) ;
        if( char_bytes(character) > utf_8_gr_eng ) { //too big to be greek or english
            for( int i = 0 ; i < char_bytes(character)-1 ; i++ ) {
                fread(&character,sizeof(unsigned char),1,src) ;
            }
            continue ;
        }
        if ( char_bytes(character) == 1) {  //english
            if( character == 10 || character == 13 || character == 20 || character == 32 ){ //spaces and newline
                fwrite(&character,sizeof(unsigned char),1,dst) ;
            }
            else if ( character > 97 && character < 122  ){ // lower case letters
                fwrite(&character,sizeof(unsigned char),1,dst) ;
            }
            else if ( character > 65 && character < 90 ){ //Upper case
                fwrite(&character,sizeof(unsigned char),1,dst) ;
            }
        }
        else {
            unsigned char nextbyte = 0 ;
            fread(&nextbyte,sizeof(unsigned char),1,src) ;
            if( character > utf_8_gr_max_first_byte) { //too big to be greek
                continue ;
            }
            if( character < utf_8_gr_min_first_byte ) { //too small to be greek
                continue ;
            }
            if( character == 206 && (( nextbyte > utf_8_gr_206_max_second) || ( nextbyte < utf_8_gr_206_min_second) || ( nextbyte == 135 )) ) { //not a letter
                continue ;
            }
            if( character == 207 && (( nextbyte > utf_8_gr_207_max_second) || ( nextbyte < utf_8_gr_207_min_second) )) { //not a letter
                continue ;
            }
            fwrite(&character,sizeof(unsigned char),1,dst) ;
            fwrite(&nextbyte,sizeof(unsigned char),1,dst) ;
        }
    }
}

unsigned short create_code(FILE* src, unsigned char character) {
    unsigned short temp ;
    if ( char_bytes(character) == 1) {
        temp = character ;
    }
    else {
        unsigned char nextbyte = 0 ;
        fread(&nextbyte,sizeof(unsigned char),1,src) ;
        temp = character ;
        temp = temp << 8 ;
        temp = temp | nextbyte ;
    }
    return temp ;
}

void decode_text(FILE* dst, FILE* src) {
    while( 1 ) {
        unsigned short temp = 0 ;
        if( feof(src) ) {
            return ;
        }
        fread(&temp,sizeof(unsigned char), 1 , src) ;
        if( temp == 0 ) {
            continue ;
        }
        if( char_bytes(temp) == 1 ){
            unsigned char tchar = (char)temp ;
            fwrite(&tchar,sizeof(unsigned char),1,dst) ;
        }
        else {
            unsigned short schar = create_code(src,temp) ;
            fwrite(&schar,sizeof(unsigned short),1,dst) ;
        }
    }
}

void encode_text(FILE* dst, FILE* src) {
    while( 1 ) {
        if ( feof(src) ) {
            break;
        }
        unsigned char character = 0 ;
        fread(&character,sizeof(unsigned char),1,src) ;
        printf("temp has value of %x \n",character) ;
        unsigned short output = create_code(src,character) ;
        fwrite(&output,sizeof(unsigned short),1,dst) ;
        if( output == 0xffff ) {
            return ;
        }
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
        if( !memcmp(argv[i],"-ec",strlen("-ec")) ) { // encode
            mode = 1 ;
        }
        else if( !memcmp(argv[i],"-dc",strlen("-dc")) ) { // decode
            mode = 2 ;
        }
        else if( !memcmp(argv[i],"-strip",strlen("-strip")) ) { // strip
            mode = 3 ;
        }
        else if( !memcmp(argv[i],"-s",strlen("-s")) ) { //source file
            i++ ;
            src = fopen(argv[i],"r") ;
            if( src == NULL) {
                printf("Failed to find source file\n") ;
                printf("Error : %s \n",strerror(errno)) ;
                return -1 ;
            }
            srcprovided++ ;
        }
        else if( !memcmp(argv[i],"-d",strlen("-d")) ) { // destination file
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
    debug() ;
    if( mode == 1 ) {
        debug() ;
        encode_text(dst,src) ;
    }
    if ( mode == 2 ) {
        debug() ;
        decode_text(dst,src) ;
    }
    if( mode == 3 ) {
        debug() ;
        fullstrip(dst,src) ;
    }
    fclose(src) ;
    fclose(dst) ;
    return 0 ;
}