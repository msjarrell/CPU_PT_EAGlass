#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <pthread.h>

#include <cuda.h>

#include "sim.h"
#include "threads.h"


void *thread_child (void *args)
{
  int id;
  float beta_low, beta_high;
  char* mydir;
  int device;
  int node;

  Arg *arg = (Arg *) args;
  id = arg->id;
  beta_low = arg->beta_low;
  beta_high = arg->beta_high;
  mydir = arg->mydir;
  device = arg->device;
  node = arg->node;

  printf ("node %03d thread %d call host_launcher (%f, %f, %s, %d, %d)\n",
	  node, id,
	  beta_low, beta_high, mydir, node, device);

  host_launcher (beta_low, beta_high, mydir, node, device);
  printf("Thread child %d,%d completed\n",node,device);
  pthread_exit (NULL);
}




void thread_parent (float beta_low, float beta_high, char *mydir, int node)
{

/*
  int ndevice;
  cudaGetDeviceCount (&ndevice);
  printf ("#GPU = %d\n", ndevice);
*/

  // replace NT with ndevice


  #define NT 2

  pthread_t threads[NT];
  Arg args[NT];
  int rc;


  for ( int t = 0; t < NT; t++) {
    args[t].id = t;
    args[t].beta_low = beta_low;
    args[t].beta_high = beta_high;
    args[t].mydir = mydir;
    args[t].device = t;
    args[t].node = node;

    rc = pthread_create(&threads[t], NULL, thread_child, (void *) &args[t]);
    PthreadCheck("pthread_create", rc);
  }

  printf("Thread parent %d before join\n",node);
  for ( int t = 0; t < NT; t++) {
    rc = pthread_join(threads[t], NULL);
    PthreadCheck("pthread_join", rc);
  }

  printf("thread parent %d completed\n",node);
}


