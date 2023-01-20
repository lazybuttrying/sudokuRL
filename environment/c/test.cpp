#include <cstdio>

extern "C" {
    int score(int arr[2]){
        return arr[0] + arr[1];
    }

    int array(double arr[3][3][1]){
        
        return arr[0][1][0] + arr[1][0][0];
    }
}