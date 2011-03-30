#include <sys/time.h>
#include <stdio.h>

inline bool get_opt(int argc, char * argv[], const char * option, int & output){
        using namespace std;
        bool opt_found=false;
        int i;
        for(i=0;i<argc;i++){
                if(argv[i][0]=='-'){
                        if(!strcmp(argv[i]+1, option)){
                                opt_found=true;
                                break;
                        }
                }
        }
        if(opt_found){
		output = strtol (argv[i+1], NULL, 10);
        }
        return opt_found;
}

double get_time(){
        cudaThreadSynchronize();
        timeval t;
        gettimeofday(&t, NULL);
        return (double)t.tv_sec+(double)t.tv_usec/1000000;
}

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        printf("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
           file, line, cudaGetErrorString( err) );
        exit(-1);
    }
}

#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
