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

#ifndef HMMCOMMUTILTEMPLATE_H
#define HMMCOMMUTILTEMPLATE_H

#include <sstream>
//=====================================
// parse command line option
//=====================================
template<class T>
inline bool get_opt(int argc, char * argv[], const char * option, T & output){
	using namespace std;
	bool opt_found=false;
	int i;
	for(i=0;i<argc;i++){
		if(argv[i][0]=='-'){
			string str(argv[i]+1);
			for(int j=0;j<str.size();j++)
				str[j]=tolower(str[j]);
			string opt(option);
			for(int j=0;j<opt.size();j++)
				opt[j]=tolower(opt[j]);
			if(str==option){
				opt_found=true;
				break;
			}
		}
	}
	if(opt_found){
		istringstream ss(argv[i+1]);
		ss>>output;
	}	
	return opt_found;
}

template<>
inline bool get_opt<bool>(int argc, char * argv[], const char * option, bool & output){
	using namespace std;
	bool opt_found=false;
	int i;
	for(i=0;i<argc;i++){
		if(argv[i][0]=='-'){
			string str(argv[i]+1);
			for(int j=0;j<str.size();j++)
				str[j]=tolower(str[j]);
			string opt(option);
			for(int j=0;j<opt.size();j++)
				opt[j]=tolower(opt[j]);
			if(str==option){
				opt_found=true;
				break;
			}
		}
	}
	if(opt_found){
		output=true;
	}	
	return opt_found;
}
#endif
