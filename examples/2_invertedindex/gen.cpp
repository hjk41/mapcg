#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>
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

	string pre="<a href=\"www.";
	string mid=".com\">";
	string post="</a>";
	int fixsize=pre.size()+mid.size()+post.size();

	srand(10);
	int fsize=0;
	while(fsize<mb){
		bool islink=rand()%100<20;	// 20 percent of lines are links
		if(islink){		
			out<<pre;
			// generate link
			string link="";
			int lsize=rand()%30;
			for(int i=0;i<lsize;i++){
				char c='a'+rand()%26;
				link+=c;
			}
			out<<link;
			out<<mid;
			// generate text
			string text="";
			int tsize=rand()%20;
			for(int i=0;i<tsize;i++){
				char c='a'+rand()%26;
				text+=c;
			}
			out<<text;
			out<<post;
			out<<endl;
			fsize+=fixsize+lsize+tsize;
		}
		else{
			int size=rand()%80;
			for(int i=0;i<size;i++){
				char c='a'+rand()%26;
				out<<c;
			}
			out<<endl;
			fsize+=size;
		}
	}

	return 0;
}
