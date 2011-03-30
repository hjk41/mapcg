/***************************************************************************

    MapCG: MapReduce Framework for CPU & GPU

    Copyright (C) 2010, Chuntao HONG (Chuntao.Hong@gmail.com).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

**************************************************************************/

#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
using namespace std;

string rand_str(){
	int num_char=rand()%6+1;
	string str(num_char,0);
	for(int i=0;i<num_char;i++)
		str[i]=rand()%26+'a';
	string result="http://www.";
	result+=str;
	result+=".com";
	return result;
}

int main(int argc, char ** argv){
	if(argc!=3){
		cout<<"usage: "<<argv[0]<<" number_MBs filename"<<endl;
		return 1;
	}

	int size=atoi(argv[1]);
	string filename=argv[2];
	cout<<"generating "<<size<<"MBs to "<<filename<<endl;

	srand(1029);
	ofstream out(filename.c_str());
	for(int i=0;i<size*1024*1024;){
		string s=rand_str();
		s+="\t 127.0.0.1";
		out<<s<<endl;
		i+=s.size()+1;
	}

	return 0;
}
