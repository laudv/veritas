import os, socket, codecs, random, time
from enum import Enum
from dask.distributed import wait, as_completed, get_client, secede, rejoin, fire_and_forget, worker_client

from .pytreeck import SearchSpace, AddTree

class DistributedVerifier:

    #class TaskType(Enum):
    #    SPLIT,
    #    VERIFY

    def __init__(self, client, addtree, verifier_factory):
        self._client = client # dask client
        self._addtree = addtree # SearchSpace
        self._verifier_factory = verifier_factory # (domains, addtree) -> Verifier
        self._branch_factor = sum(client.nthreads().values()) * 4

    #def run(self):
    #    c = self._client
    #    timeout = 60
    #    futures0 = [c.submit(DistributedVerifier._split_fun,
    #        None, self._addtree, self._branch_factor)]
    #    futures1 = []

    #    # maybe work with handlers instead of as_completed
    #    while len(futures0) > 0:
    #        for future in as_completed(futures0):
    #            tup = future.result()
    #            typ = tup[0]
    #            if typ == DistributedVerifier.TaskType.SPLIT:
    #                futures1 += self._handle_split_result(*tup[1:], timeout)
    #            elif typ == DistributedVerifier.TaskType.SPLIT:
    #                futures1 += self._handle_verify_result(*tup[1:])
    #            else:
    #                raise RuntimeError("unknown TaskType")

    #        timeout *= 1.2
    #        futures0 = futures1
    #        futures1 = []

    def run(self):
        #fs = [self._client.submit(DistributedVerifier.test2, 1) for i in range(10)]

        #for f in as_completed(fs):
        #    print("[4]   bye from", f.result(), os.getpid() % 10)

        fs = [self._client.submit(DistributedVerifier.test2, 1)]

        while len(fs) > 0:
            print(len(fs), "=================")
            wait(fs, return_when="FIRST_COMPLETED")
            next_fs = []
            for f in fs:
                if f.done():
                    i, new_fs = f.result()
                    if new_fs is not None:
                        if isinstance(new_fs, list):
                            print("list", i)
                            for new_f in new_fs:
                                next_fs.append(new_f)
                        else:
                            print("single", i)
                            next_fs.append(new_fs)
                    else:
                        print("done", i)
                else:
                    next_fs.append(f)
            fs = next_fs
        print("done!")

    @staticmethod
    def test2(i):
        print("hallo!", i, os.getpid() % 10)
        time.sleep(1.0)
        if i < 10:
            client = get_client()
            if i % 2 == 1:
                return i, client.submit(DistributedVerifier.test2, i+1)
            else:
                return i, [client.submit(DistributedVerifier.test2, i+j+1) for j in range(2)]
        return i, None

    #def run(self):
    #    print("hello??")
    #    print("running", os.getpid())
    #    at_enc = DistributedVerifier._enc_at(self._addtree)
    #    f = self._submit_split_task(None, at_enc)
    #    #print("result: ", f.result())

    #def _submit_split_task(self, doms, at_enc):
    #    future = self._client.submit(DistributedVerifier._split_fun,
    #            doms, at_enc, self._branch_factor)
    #    future.add_done_callback(lambda f: print("hello world!"))
    #    return future

    #def _handle_split_result(future):
    #    print("hello world! no pid?", os.getpid() % 10)
    #    futures = []
    #    print("hoi", os.getpid()% 10)
    #    for doms, at_enc in future.result():
    #        print("hello?")
    #        print(future, doms)
    #        #f = self._client.submit(DistributedVerifier._verify_fun,
    #        #        self._verifier_factory, doms, at_enc
    #        #futures.append(f)
    #    return futures

    #def _handle_verify_result(status):
    #    pass

    #def _split_and_verify(self, addtree, domains=None):
    #    sp = SearchSpace(self.addtree)
    #    sp.split(self._branch_factor)

    #    futures = []
    #    for leaf_id in sp.leafs():
    #        addtree = sp.get_pruned_addtree(leaf_id).to_json()
    #        domains = sp.get_domains(leaf_id)
    #        future = self.client.submit(DistributedVerifier._solve,
    #                self._verifier_factory,
    #                leaf_id, domains, addtree, 60)
    #        futures.append(future)

    #    for future in as_completed(futures):
    #        y = future.result()
    #        print("RESULT as_completed", y, socket.gethostname(), os.getpid(), "done")

    def _split_fun(domains, addtree, branch_factor):
        addtree = DistributedVerifier._dec_at(addtree)
        if domains is None: sp = SearchSpace(addtree)
        else:               sp = SearchSpace(addtree, domains)

        sp.split(branch_factor)

        results = []
        for leaf_id in sp.leafs():
            doms = sp.get_domains(leaf_id)
            at = sp.get_pruned_addtree(leaf_id)
            at = DistributedVerifier._enc_at(at)
            results.append((doms, at))

        return results

    def _verify_fun(vfactory, domains, addtree, timeout):
        addtree = DistributedVerifier._dec_at(addtree)
        v = vfactory(domains, addtree)
        v.set_timeout(timeout)
        status = v.verify() # maybe also return logs, stats
        return status

    def _enc_at(at):
        b = bytes(at.to_json(), encoding="ascii")
        return codecs.encode(b, encoding="zlib")

    def _dec_at(b):
        at_json = codecs.decode(b, encoding="zlib").decode("ascii")
        return AddTree.from_json(at_json)



