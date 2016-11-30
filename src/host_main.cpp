#include "COPYING"


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "sim.h"
//#include "threads.h"
#include "mpiprocess.h"

//#include "hdf5.h"


//extern "C"
//{
void mpiprocess (float beta_low, float beta_high, char *mydir);
//}

/*
void
mpiprocess (float beta_low, float beta_high, char *mydir)
{
  printf ("mpiprocess: hello\n");
}
*/


// ./a.out [-l beta_low] [-u beta_high] [-o output]
int
main (int argc, char **argv)
{
  float beta_low = BETA_LOW;
  float beta_high = BETA_HIGH;
  char mydir[STR_LENG] = "output_default";


  // parse command line parameters
  if (argc != 1)
    for (int i = 1; i < argc; i++) {
      if (argv[i][0] == '-') {	// leading "-"
	switch (argv[i][1]) {
	case 'l':
	  beta_low = strtof (argv[++i], NULL);
	  break;
	case 'u':
	  beta_high = strtof (argv[++i], NULL);
	  break;
	case 'o':
	  strcpy (mydir, argv[++i]);
	  break;
	default:
	  host_usage (argv[0]);
	}
      }
      else
	host_usage (argv[0]);
    }
  if (beta_low >= beta_high)
    host_usage (argv[0]);


  // obtain the time tag
  char mystime[STR_LENG];
  time_t mytime = time (NULL);
  struct tm *mylocaltime;
  mylocaltime = localtime (&mytime);
  strftime (mystime, STR_LENG, "%Y%m%d_%H%M%S", mylocaltime);

  // name the output directory using time tag
  if (strcmp (mydir, "output_default") == 0){
    //    sprintf (mydir, "output_%s", mystime);
    int l=SZ;
    float h=H;
    float beta_low=BETA_LOW;
    float beta_high= BETA_HIGH;
    sprintf(mydir,"L%dH%1.2f_%1.2f_%1.2f_%s",l,h,beta_low,beta_high,mystime);
  }

#ifndef NO_OUTPUT
    host_makedir (mydir);
#endif

  //mylocaltime = localtime (&mytime);
  //printf ("start simulation\t %s", asctime (mylocaltime));

  int node = 0;
  int gpu = 0;
  //printf ("host_main: hello\n");

  //host_launcher (beta_low, beta_high, mydir, node, gpu);
  //thread_parent (beta_low, beta_high, mydir, node);
  mpiprocess (beta_low, beta_high, mydir);

  //mylocaltime = localtime (&mytime);
  //printf ("finished simulation\t %s", asctime (mylocaltime));
  host_summary (beta_low, beta_high, mydir);

  return 0;
}

