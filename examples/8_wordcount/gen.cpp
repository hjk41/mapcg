#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

int main(int argc, char ** argv){
	if(argc!=3){
		cout<<"usage: "<<argv[0]<<" number_of_MB file_name"<<endl;
		return 1;
	}

	int mb=atoi(argv[1]);
	cout<<"generating "<<mb<<" MBs to "<<argv[2]<<endl;
	mb*=1024*1024;
	ofstream out(argv[2]);

	srand(10);
	int fsize=0;
	int lsize=0;
	while(fsize<mb){
		int wsize=rand()%10;
		for(int i=0;i<wsize;i++){
			char c='a'+rand()%26;
			out<<c;
		}
		out<<" ";
		lsize+=wsize+1;
		if(lsize>=80){
			out<<endl;
			lsize=0;
			fsize+=1;
		}
		fsize+=wsize+1;
	}

	return 0;
}
