#include <stdlib.h>
#include "lcg.h"
#include "sim.h"



void
host_init_seed (LCG_DATATYPE *gpuseed)
{
  for (int i = 0; i < TperB * GD; i++)
    gpuseed[i] = (LCG_DATATYPE) rand ();
}


