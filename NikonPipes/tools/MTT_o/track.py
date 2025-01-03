class Track:
    def __init__(self,min_count,max_count,ide):

        self.tentative = True
        self.killed = False
        self.min_count = min_count
        self.max_count = max_count
        self.count = 0
        self.count_associated = 0
        self.missed = 0
        self.history = []
        self.status = []
        self.id = ide
        self.indices = []

    def update(self,z=None):
        if z is not None:
            self.update_counters(True)
        else:
            self.update_counters(False)

    def update_counters(self,associated):
        """ Update the number of associated measurements
            Switch state between alive/tentative/killed

            TODO: k/n initialization from tentative to alive

            @associated: True/False if there is a measurement
        """
        if not self.killed:
            self.count += 1
            if associated:
                self.missed = 0
                self.count_associated += 1
            else:
                self.missed += 1

            if self.tentative:
                # if enough measurements associated
                if self.count_associated>self.min_count:
                    self.tentative = False

            # if we have missed max_count amount of measurements
            if self.missed>self.max_count:
                # change to tentative or killed
                if self.tentative:
                    self.killed = True
                else:
                    # kill
                    self.killed = True
                    self.tentative = True
                    self.count_associated = 0
                    # should perhaps add separate tentative threshold
                    self.missed = 0

    @staticmethod
    def associate(trackers,measurements):
        pass




