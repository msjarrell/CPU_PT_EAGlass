#ifndef THREADS_H
#define THREADS_H




#define COND_CHECK(func, cond, retv, errv)				\
  if ( (cond) )								\
    {									\
      fprintf(stderr, "\n[CHECK FAILED at %s:%d]\n| %s(...)=%d (%s)\n\n", \
	      __FILE__, __LINE__, func, retv, strerror(errv));		\
      exit(EXIT_FAILURE);						\
    }
#define ErrnoCheck(func, cond, retv)  COND_CHECK(func, cond, retv, errno)
#define PthreadCheck(func, rc) COND_CHECK(func,(rc!=0), rc, rc)





struct Arg
{
  int id;

  float beta_low;
  float beta_high;
  char* mydir;
  int device;
  int node;
};


void *thread_child (void *args);
void thread_parent (float beta_low, float beta_high, char *mydir, int node);


#endif

