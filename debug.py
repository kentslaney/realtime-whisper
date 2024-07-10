import pdb, pathlib, sys

class Cwdb(pdb.Pdb):
    zoning = False
    def do_zoned(self, arg):
        """z(oned)

        Execute the current line, stop at the first possible occasion inside a
        file directly contained by the current working directory.
        """
        if arg:
            self._print_invalid_arg(arg)
            return
        self.zoning = True
        self._set_stopinfo(None, None)
        return 1
    do_z = do_zoned

    def stop_here(self, frame):
        if self.zoning:
            if not self.relevant(frame):
                return False
            self.zoning = False
        return super().stop_here(frame)

    def relevant(self, frame):
        filename = self.canonic(frame.f_code.co_filename)
        return pathlib.Path.cwd().samefile(pathlib.Path(filename).parents[0])

def set_trace(*, header=None):
    pdb = Cwdb()
    if header is not None:
        pdb.message(header)
    pdb.set_trace(sys._getframe().f_back)

sys.breakpointhook = set_trace

