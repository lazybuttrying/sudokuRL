#include <cstdio>

#define MAX_SIZE 4
#define BOARD_BASE 2

extern "C" {
    int calc_score(int sudoku[MAX_SIZE][MAX_SIZE][1]) {
        int score = 0;
        
        int i=0, j=0;
        int rtotal[MAX_SIZE+1] = {0,};
        int ctotal[MAX_SIZE+1] = {0,};
        int ztotal[MAX_SIZE+1] = {0,};
        int rsum=0, csum=0, zsum=0;

        for (i = 0; i < MAX_SIZE; i++) {
            for (j = 0; j < MAX_SIZE; j++) {
                if (sudoku[i][j][0])
                    rtotal[sudoku[i][j][0]] = 1;
                if (sudoku[j][i][0])
                    ctotal[sudoku[j][i][0]] = 1;                
            }
            
            rsum=0;
            csum=0;

            for (j = 1; j <= MAX_SIZE; j++) {
                rsum += rtotal[j];
                csum += ctotal[j];
                rtotal[j] = 0;
                ctotal[j] = 0;
            }

            if (rsum == MAX_SIZE)
                score += 1;
            if (csum == MAX_SIZE)
                score += 1;
        }

        for (i=0; i<BOARD_BASE; i++) {
            for (j=0; j<BOARD_BASE; j++) {

                for (int k=0; k<BOARD_BASE; k++) {
                    for (int l=0; l<BOARD_BASE; l++) {
                        if (sudoku[i*BOARD_BASE+k][j*BOARD_BASE+l][0])
                            ztotal[sudoku[i*BOARD_BASE+k][j*BOARD_BASE+l][0]] = 1;
                    }
                }

                zsum = 0;
                for (int k=1; k<=MAX_SIZE; k++) {
                    zsum += ztotal[k];
                    ztotal[k]=0;
                }
                if (zsum == MAX_SIZE)
                    score += 1;
                    
            }
        }

        return score;
    }
}