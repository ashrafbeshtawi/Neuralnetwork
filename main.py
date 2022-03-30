import _thread
import visual
import mutate


shared_Neural_Network = {}
_thread.start_new_thread( mutate.muatete_run, ( shared_Neural_Network, ) )
visual.runGraph(shared_Neural_Network)



