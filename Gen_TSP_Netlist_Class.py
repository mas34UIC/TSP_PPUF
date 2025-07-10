import numpy as np
import csv
import os

'''
Class to generate a netlist for simulating TSP CMOS ameboid based solver from: https://doi.org/10.1038/s41598-020-77617-7
'''

class Gen_Netlists:

    def __init__(self):
        self.n = None
        self.netlist = ""
        self.adj_matrix = []
        self.lambd = None

    def init_netlist(self):
        # define current source
        self.netlist += f"* Current source\n"
        i = 5*(self.n*self.n)
        self.netlist += f"I1 0 Ni PULSE(0 {i}u 50u 1u 1u 1 2)\n\n"
        # define pseudopod op-amp voltage, used as voltage follower/buffer
        self.netlist += f"* Pseudopod op-amp supply voltage\n"
        self.netlist += "Vdd1 Vdd 0 4\n\n"
        # define IMC op-amp voltages
        self.netlist += f"* IMC op-amp supply voltages\n"
        self.netlist += "Vcc1 Vcc 0 15\n"
        self.netlist += "Vcc2 0 -Vcc 15\n"
        self.netlist += "Vcc3 VL 0 5\n\n"
        self.netlist += "VRef 0 VT 1.5\n\n"
        

    def generate_branches(self):
        # generate n^2 variable "pseudopod" branches
        self.netlist += f"* add n^2 variable pseudopod branches\n\n"
        for i in range(1, self.n+1):
            for j in range(1, self.n+1):
                self.netlist += f"* Branch {i:02d}_{j:02d}\n"
                # input resistor
                self.netlist += f"RB{i:02d}_{j:02d}1 Ni NB{i:02d}_{j:02d}1 {round(np.random.uniform(1,10000),4)}\n"
                # add diode
                self.netlist += f"D{i:02d}_{j:02d} NB{i:02d}_{j:02d}1 NB{i:02d}_{j:02d}2 D_custom\n"
                # add cap and nmos in ||
                #C = round(500 + 500*(np.random.uniform(-0.01, 0.01)), 4)
                C = 500
                self.netlist += f"C{i:02d}_{j:02d} NB{i:02d}_{j:02d}2 0 {C}p ic=1.5\n"
                self.netlist += f"M{i:02d}_{j:02d} NB{i:02d}_{j:02d}2 L{i:02d}_{j:02d} 0 0 NMOS_custom w=1u l=1u\n" 
                # add voltage follower buffer
                self.netlist += f"X§amp{i:02d}_{j:02d}1 NB{i:02d}_{j:02d}2 NB{i:02d}_{j:02d}3 Vdd 0 NB{i:02d}_{j:02d}3 level2 Avol=1Meg GBW=10Meg Slew=10Meg Ilimit=25m Rail=0 Vos=0 En=0 Enk=0 In=0 Ink=0 Rin=500Meg\n"
                # add resistors and inverter
                self.netlist += f"RB{i:02d}_{j:02d}2 NB{i:02d}_{j:02d}3 NB{i:02d}_{j:02d}4 390k\n"
                self.netlist += f"RB{i:02d}_{j:02d}3 NB{i:02d}_{j:02d}4 NB{i:02d}_{j:02d}5 2.2Meg\n"
                self.netlist += f"X§INV{i:02d}_{j:02d} NB{i:02d}_{j:02d}4 NB{i:02d}_{j:02d}5 Inv\n"
                # add final voltage follower buffer
                self.netlist += f"X§amp{i:02d}_{j:02d}2 NB{i:02d}_{j:02d}5 X{i:02d}_{j:02d} Vdd 0 X{i:02d}_{j:02d} level2 Avol=1Meg GBW=10Meg Slew=10Meg Ilimit=25m Rail=0 Vos=0 En=0 Enk=0 In=0 Ink=0 Rin=500Meg\n\n"
                # for testing purposes---------
                # self.netlist += f"RB{i}{j}4 X{i}{j} 0 1k\n"
                # self.netlist += f"V{i}{j} L{i}{j} 0 PULSE(0 5 0 1u 1u 100u 200u)\n\n"
        
    def gen_step_function(self):
        self.netlist += f"* add n^2 bounceback control outputs\n\n"

        # add n^2 "bounceback" control outputs 
        for i in range(1, self.n+1):
            for j in range(1, self.n+1):
                self.netlist += f"* L{i:02d}_{j:02d}\n"
                # add inv summing op-amp 
                self.netlist += f"X§U{i:02d}_{j:02d}1 0 LIN{i:02d}_{j:02d} Vcc -Vcc NLS{i:02d}_{j:02d} level2 Avol=1Meg GBW=10Meg Slew=10Meg Ilimit=25m Rail=0 Vos=0 En=0 Enk=0 In=0 Ink=0 Rin=500Meg\n"
                self.netlist += f"RL{i:02d}_{j:02d}1 LIN{i:02d}_{j:02d} NLS{i:02d}_{j:02d} 10k\n"
                # add comparator
                self.netlist += f"X§U{i:02d}_{j:02d}2 NLS{i:02d}_{j:02d} VT VL 0 L{i:02d}_{j:02d} level2 Avol=1Meg GBW=10Meg Slew=10Meg Ilimit=25m Rail=0 Vos=0 En=0 Enk=0 In=0 Ink=0 Rin=500Meg\n\n"
                # for testing ----------
                # self.netlist += f"V4 NL{i}{j}5 0 SINE(3 1 0.5)\n"
                # self.netlist += f"RL{i}{j}5 NL{i}{j}5 LIN{i}{j} 20k\n"
                # self.netlist += f"RL{i}{j}6 L{i}{j} 0 1k\n\n"

    def gen_IMC(self):
        self.netlist += "* Crossbar IMC\n"
        
        for v in range(1, self.n+1):
            for k in range(1, self.n+1):
                self.netlist += f"\n* Connections to output X{v:02d}_{k:02d} node\n"
                for u in range(1, self.n+1):
                    for l in range(1, self.n+1):
                        if (v == u and k != l) or (v != u and k == l):
                            self.netlist += f"RI{v:02d}{k:02d}{u:02d}{l:02d} X{v:02d}_{k:02d} LIN{u:02d}_{l:02d} 20k\n"
                        elif (v != u) and (abs(k-l) == 1):
                            dist_vu = self.adj_matrix[v-1][u-1]
                            R = round(10000*((self.lambd/0.2)/dist_vu), 4)
                            #R = R + round((R*np.random.uniform(-0.01, 0.01)), 4)
                            self.netlist += f"RI{v:02d}{k:02d}{u:02d}{l:02d} X{v:02d}_{k:02d} LIN{u:02d}_{l:02d} {R}\n"
                                            
    def end_list(self):
        # add analysis
        self.netlist += "* add analysis\n"
        self.netlist += "\n.tran 0 2m 0 500n\n\n"
        self.netlist += "* block symbol definitions\n"
        self.netlist += ".subckt Inv In Out\n"
        self.netlist += "M1 Out In 0 0 N_1u l=1u w=10u\n"
        self.netlist += "M2 Vdd In Out Vdd P_1u l=1u w=28u\n"
        self.netlist += "V1 Vdd 0 4\n"

        '''
        Modify with path of your bsim file
        '''
        self.netlist += r".include C:\Users\stanm\Documents\LTspice\SpiceComponents\bsim.txt" + "\n"
        self.netlist += ".ends Inv\n\n"
        
        '''
        I is used to limit the current that can pass through the diode, so that only valid solution states can be reached.
        A more realistic current limiting circuit could be used, but I went with this for simplicity.
        '''
        I = round(((5*(self.n*self.n))/(self.n))+1,1)
        self.netlist +=f".model D_custom D(Ilimit={I}u Ron=0 Vfwd=0.7 Epsilon=0.1)\n"
        
        '''
        Must be modified with the path of your standard diode and mosfet files from ltspice.
        This path can be found by placing a diode and mosfet into an ltspice schematic, and looking at the netlist view.
        '''
        self.netlist += r".lib C:\Users\stanm\AppData\Local\LTspice\lib\cmp\standard.dio" + "\n"
        self.netlist += ".model NMOS_custom NMOS(Vto=0.4)\n"
        self.netlist += ".model NMOS NMOS\n"
        self.netlist += r".lib C:\Users\stanm\AppData\Local\LTspice\lib\cmp\standard.mos" + "\n"
        self.netlist += ".lib UniversalOpAmp2.lib\n"
        self.netlist += ".backanno\n"
        self.netlist += ".end"
        
    def read_distance_matrix_from_csv(self, filename):
        with open(filename, newline='') as f:
            reader = list(csv.reader(f))
            self.n = int(reader[0][1])
            self.lambd = float(reader[1][1])
            matrix_rows = reader[9:9+self.n]
            for row in matrix_rows:
                self.adj_matrix.append([float(x) for x in row[1:]])

    def gen_netlist(self, paths):
        for path in paths:
            self.netlist = ""
            self.adj_matrix.clear()
            self.read_distance_matrix_from_csv(path)
            self.init_netlist()
            self.generate_branches()
            self.gen_step_function()
            self.gen_IMC()
            self.end_list()
            file = os.path.join(os.path.dirname(path), f"generated_{self.n}_node.cir")
            with open(file, "w") as f:
                f.write(self.netlist)
            print(f"Generated Netlist IMC with {self.n} nodes and saved to {file}\n")
