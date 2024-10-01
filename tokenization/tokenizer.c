#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<errno.h>
#define debug() printf("Debug %d\n",__LINE__)

void print_table( int* table, int*table_size) {
    printf("Table has size %d and the following elements \n",*table_size) ;
    for( int i = 0 ; i < *table_size ; i++ ) {
        printf(" Element %d has index of %d \n", table[i],i) ;
    }
}

void print_corpus( int* corpus, unsigned long long corpus_size ) {
    printf("Corpus is %llu size and the following text: \n",corpus_size) ;
    for( unsigned long long i = 0 ; i < corpus_size ; i++ ) {
        printf("%d ",corpus[i]) ;
    }
    printf("\n") ;
}

void expand_corpus_size(int** corpus, unsigned long long* corpus_size) {
    corpus[0] = realloc(corpus[0],sizeof(int)*(*corpus_size+1)) ;
    (*corpus_size)++ ;
    return ;
}

void expand_table_size(int** table, int* table_size ) {
    table[0] = realloc(table[0],sizeof(int)*(*table_size+1)) ;
    (*table_size)++ ;
    return ;
}

void expand_table_init( int** table, int* table_size, short temp ) {
    for( int i = 0 ; i < *table_size ; i++ ) {
        if( (int)temp == table[0][i] ) {
            return ;
        }
    }
    if( temp == 0 ) {
        return ;
    }
    expand_table_size(table,table_size) ;
    table[0][(*table_size-1)] = (int) temp ;
    return ;
}

void begin_constructing_table(FILE* src,int** table, int* table_size ) {
    while(1) {
        if( feof(src) ) {
            break ;
        }
        unsigned short temp ;
        fread(&temp,sizeof(unsigned short),1,src) ;
        expand_table_init(table,table_size,temp) ;
    }
    return ;
}

void load_text (FILE* src, int* table, int table_size, int** corpus, unsigned long long* corpus_size) {
    while(1) {
        if( feof(src) ) {
            break ;
        }
        short temp ;
        fread(&temp,sizeof(short),1,src) ;
        for( int i = 0 ; i < table_size ; i++ ) {
            if( (int)temp == table[i] ) {
                expand_corpus_size(corpus,corpus_size) ;
                corpus[0][*corpus_size-1] = i ;
                break ;
            }
        }
    }
    return ;
}

int main(int argc, char** argv) {
    //FILE* dst = fopen(argv[1],"w") ;
    //if( dst == NULL ) {
    //    printf("Failed to find or create destination file\n") ;
    //    printf("Error : %s \n",strerror(errno)) ;
    //    return -1 ;
    //}
    int* table = NULL ;
    unsigned int table_size = 0 ;
    int* corpus = NULL ;
    unsigned long long corpus_size = 0 ;
    FILE* src = fopen(argv[2],"r") ;
    if( src == NULL) {
        printf("Failed to find source file\n") ;
        printf("Error : %s \n",strerror(errno)) ;
        return -1 ;
    }
    begin_constructing_table(src,&table,&table_size) ;
    //print_table(table,&table_size) ;
    rewind(src) ;
    load_text(src,table,table_size,&corpus,&corpus_size) ;
    print_corpus(corpus,corpus_size) ;
    return 0 ;
}