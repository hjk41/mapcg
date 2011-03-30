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

#ifndef HMMTSQUEUE_H
#define HMMTSQUEUE_H

//----------------------------------
// Threads-Safe Queue
//----------------------------------

#include <pthread.h>
#include <queue>

template<class T>
class TSQueue{
public:
	TSQueue(){
		pthread_mutex_init(&_lock,NULL);
	}
	~TSQueue(){
		pthread_mutex_destroy(&_lock);
	}

	void push(const T & e){
		pthread_mutex_lock(&_lock);
		_queue.push(e);
		pthread_mutex_unlock(&_lock);
	}

	bool pop(T & e){
		bool success=true;
		pthread_mutex_lock(&_lock);
		if(!_queue.empty()){
			e=_queue.front();
			_queue.pop();
			success=true;
		}
		else{
			success=false;
		}
		pthread_mutex_unlock(&_lock);

		return success;
	}
private:
	std::queue<T> _queue;
	pthread_mutex_t _lock;
};

#endif
