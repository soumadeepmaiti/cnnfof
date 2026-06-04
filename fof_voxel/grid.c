#include "allvars.h"
#include "proto.h"
#include <algorithm>

void Sort_part()
{
  clock_t t = clock();

  NG = 1 << BITS;
  LGRID = LBOX/(double)NG;
  fprintf(stdout," | NG = %d \n",NG);
  fprintf(stdout," | Lado del grid = %6.3f [Mpc/h] \n",LGRID);
  #pragma omp parallel for
  for (int p=0; p<NTRAC; p++) {

      int i = (int)(Tracer[p].Pos[0]/LGRID);
      int j = (int)(Tracer[p].Pos[1]/LGRID);
      int k = (int)(Tracer[p].Pos[2]/LGRID);
      Tracer[p].pkey= peano_hilbert_key(i,j,k);
  }

  std::sort(Tracer, Tracer + NTRAC, [](const tracers& a, const tracers& b) { return a.pkey < b.pkey; });

  Time(t,"Sort");
}

int peano_hilbert_key(int x, int y, int z)
{
  int i, quad, bitx, bity, bitz;
  int mask, rotation, rotx, roty, sense;
  int key;


  mask = 1 << (BITS - 1);

  key = 0;
  rotation = 0;
  sense = 1;


  for(i = 0; i < BITS; i++, mask >>= 1)
    {
      //printf("mask  %d\n",mask);
      bitx = (x & mask) ? 1 : 0;
      bity = (y & mask) ? 1 : 0;
      bitz = (z & mask) ? 1 : 0;
      
      //printf("x    %d y    %d z    %d\n",x,y,z);
      //printf("bitx %d bity %d bitz %d\n",bitx,bity,bitz);

      quad = quadrants[rotation][bitx][bity][bitz];
      //printf("rotation %d quad %d\n",rotation,quad);

      key <<= 3;
      //printf("keya %d\n",key);
      //printf("sense %d\n",sense);
      key += (sense == 1) ? (quad) : (7 - quad);
      //printf("keyb %d\n",key);

      rotx = rotx_table[quad];
      roty = roty_table[quad];
      sense *= sense_table[quad];
      //printf("sense*table  %d\n",sense);

      while(rotx > 0)
	{
	  rotation = rotxmap_table[rotation];
	  rotx--;
	}

      while(roty > 0)
	{
	  rotation = rotymap_table[rotation];
	  roty--;
	}
      //printf("rotx %d roty %d \n",rotx,roty);
    }

  return key;
}

void Grid()
{
  int     p,i,j,k,l;
  clock_t t;

  fprintf(stdout,"\n Armando linked-list\n");fflush(stdout);
  t = clock();
 

  Cabecera = (int *) malloc(NG*NG*NG*sizeof(int));
  Linklist = (int *) malloc(NTRAC*sizeof(int));

  for (p=0; p<NG*NG*NG; p++) 
      Cabecera[p] = -1;

  for (p=0; p<NTRAC; p++) 
      Linklist[p] = -1;

  for (p=0; p<NTRAC; p++) {

      i = (int)(Tracer[p].Pos[0]/LGRID);
      j = (int)(Tracer[p].Pos[1]/LGRID);
      k = (int)(Tracer[p].Pos[2]/LGRID);

      l = (i*NG + j)*NG + k;

      Cabecera[l] = p;
  }

  for (p=0; p<NTRAC; p++) {

      i = (int)(Tracer[p].Pos[0]/LGRID);
      j = (int)(Tracer[p].Pos[1]/LGRID);
      k = (int)(Tracer[p].Pos[2]/LGRID);

      l = (i*NG + j)*NG + k;

      Linklist[Cabecera[l]] = p; 
      Cabecera[l] = p;
  }

  Time(t,"Grid");

}

int iper(int i)
{
  double ip;
	
  if (i >= NG) {
     ip = i - NG;	  
  } else if (i < 0) {
     ip = i + NG;	  
  } else {
     ip = i;	  
  }

  return ip;
}

void FreeGrid()
{
   free(Cabecera);
   free(Linklist);
}


