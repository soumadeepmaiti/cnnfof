
#include "allvars.h"
#include "proto.h"

int CountLines(const char *filename)
{
  FILE *fp;
  int  count = 0;
  char string[256];

  fp = fopen(filename, "r");
  if (fp == NULL) {
    fprintf(stdout,"ERROR!! %s no existe \n",filename);
    exit(EXIT_FAILURE);
  }

  while (fgets(string,256,fp)) count++;
  fclose(fp);

  return count; 
}
    
void Time(clock_t ti, const char* msg)
{
  clock_t tf;
  float   tseg, tmin;

  tf = clock();

  tseg = (float)(tf - ti) / CLOCKS_PER_SEC;
  tmin = tseg / 60.0;

  fprintf(stdout, " | %s -> tiempo = %f min.\n", msg, tmin);
}

void Progress(int par, int tot)
{
  float prog;

  if (par%(int)((float)tot/100.0) == 0) { 
     prog = (float)par/(float)tot*100.0;
     fprintf(stdout," | %d %s \r",(int)prog,"%");
     fflush(stdout);
  }

}
