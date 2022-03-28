from multiprocessing import Process
import visual
import mutate



if __name__ == '__main__':
    p = Process(target = visual.runGraph)
    w = Process(target = mutate.muatete_run)
    p.start()
    w.start()
    w.join()
    p.join()