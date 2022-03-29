import _thread
import visual
import mutate


shared_list = []
_thread.start_new_thread( mutate.muatete_run, ( shared_list, ) )
visual.runGraph(shared_list)
#mutate.muatete_run(shared_list)


