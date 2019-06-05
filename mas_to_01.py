"""
A simple multi-agent evolution design demo

TODO:

knowlegment presentation
rule deduce
decision and action




Reference:
1. CLIPs http://pyclips.sourceforge.net/web/
2. Life game https://github.com/electronut/pp/blob/master/conway/conway.py
3. pyKnow https://pyknow.readthedocs.io/en/stable/introduction.html

"""
import sys, argparse
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


ON = 255
OFF = 0
vals = [ON, OFF]

'''
Tool funcs
'''
def lk(nu=0.3):
    E=1

    k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
    return (KE)

def preFilter(x,y,rmin = 1.2):
    
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    nfilter=x*y*((2*(np.ceil(rmin)-1)+1)**2)
    nfilter = int(nfilter)
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc=0
    for i in range(x):
        for j in range(y):
            ro=i*y+j
            kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
            kk2=int(np.minimum(i+np.ceil(rmin),x))
            ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
            ll2=int(np.minimum(j+np.ceil(rmin),y))
            for k in range(kk1,kk2):
                for l in range(ll1,ll2):
                    co=k*y+l
                    fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                    iH[cc]=ro
                    jH[cc]=co
                    sH[cc]=np.maximum(0.0,fac)
                    cc=cc+1
    # Finalize assembly and convert to csc format
    H=coo_matrix((sH,(iH,jH)),shape=(x*y,x*y)).tocsc()    
    Hs=H.sum(1)
    return H, Hs
'''
rule:

In order to bridge the mathematical variables and logical rules, We must design a scheme that 
describe "rules" in text mode and deduce if the current variables are match with the rules
during the run time. That means the data of program and knowlegement are seperated. A knowlege
engine should be used to apply the knowlege to the data.

example:
   [rules]
   (total < 2 or total > 3) and state = ON -> state=OFF
   (total == 3) -> state=ON
   
   [facts/variables]
   total = 3
   status = X
   
How to make a decision by state varibles and rules?

The solution is:
- use @decrator to define fact and rules[easy to formulate]
- use sensor to convert varible and value to fact[eg. sensor(total)]
   
example:
    ke = KnowlegeEngine()
    ke.add_fact(total= 2)
    ke.add_fact(state=ON)
    @Rule(
      AND(
        OR(
        TEST(total, lambda x: x < 2),
        TEST(total, lambda x, > 3)
        ),
        Fact(state = ON)
      )
      agent.act_xxx:
          pass
    
   
'''

class KnowlegeEngine():
    '''    
    1. Hold an knowlege base (eg. rules and facts) for application
    2. Help agents to make a decision.
    '''
    def __init__(self):
        self.rules = {}
        self.facts = {}
        return
    def reset(self):
        '''
        Unlike with CLIPs, our knowlege engine must hold some consistant status, so 
        we prefer to adopt "update" instead of "reset" 
        '''
        return
    
    def add_rules(self, *rule):
        '''
        f(agent,env)
        '''
        idx = len(self.rules.keys())
        rules[idx] = rule
        return
    def add_fact(self, **kwargs):
        f = dict(kwargs)
        self.facts.update(f)
        return
    def inference(self):
        
        return
    def status(self):
        print("Facts:")
        i = 0
        for k, v in self.facts.items():
            msg = "<f-{}> {} = {}".format(i, k, v)
            print(msg)
            i += 1




ke = KnowlegeEngine()
class Fact():
    '''
    Test version of Fact
    ''' 
    def __init__(self, **kwargs):
        global ke
        ke.add_fact(**kwargs)
   
class Rule(tuple):
    '''
    Test version of Rule
    '''
    def __init__(self, **kwarg):
        global ke
        ke.add_rules()
        return
    def __call__(self, *args, **kwargs):
        
        pass
        
class Agent:
    def __init__(self, ke, env):
        #set global KB engine and enviorment
        self.ke = ke
        self.env = env
        
        self.pos = [0,0]
        self.total = 0
        
        return
    def bind_pos(self, pos):
        self.pos = pos
    def sense(self, grid):
        # get knowlegement about environment
        i = self.pos[0]
        j = self.pos[1]
        [H, W] = grid.shape
        total = int((grid[i, (j-1)%W] + grid[i, (j+1)%W] + 
                             grid[(i-1)%H, j] + grid[(i+1)%H, j] + 
                             grid[(i-1)%H, j] + grid[(i-1)%H, (j+1)%W] + 
                             grid[(i+1)%H, (j-1)%W] + grid[(i+1)%H, (j+1)%W])/255)        
        return total, grid[i,j] 
    def make_decision(self, **fact):
        # make decisions and do actions by current knowledge
        facts = dict(**fact)
        ##(total < 2 or total > 3) and state = ON -> state=OFF
        ##(total == 3) -> state=ON
        if (facts.get('total') < 2 or facts.get('total') > 3 ) and (facts.get('state') == ON):
            self.act_OFF()
        if (facts.get('total') == 3 ):
            self.act_ON()
        
        return
    
    def act(self,grid):
        total, state = self.sense(grid)   # todo: bind sense to fact
        self.make_decision(total = total, state = state)
        return
    def act_OFF(self):
        pos = self.pos
        self.env.grid[pos[0],pos[1]] = OFF 
    def act_ON(self):
        pos = self.pos
        self.env.grid[pos[0],pos[1]] = ON     
   
      

class Environment:
    def __init__(self):
        self.grid = np.zeros(0)
        ## properties list
        self.grid_prop_ce= np.zeros(0)
        self.grid_prop_dc= np.zeros(0)
        ##self.grid_prop_stress= np.zeros(0)
        self.step = 0
        return
    def set_dim(self, row,col,height=0):
        '''
        init a grid with dimention of (row, col, height)
        '''
        self.L = row
        self.W = col
        self.ndof = 2*(col+1)*(row+1)
        
        ### init constraints
        # F, Fix, Support
        self.edofMat=np.zeros((col*row,8),dtype=int)
        for elx in range(col):
            for ely in range(row):
                el = ely+elx*row
                n1=(row+1)*elx+ely
                n2=(row+1)*(elx+1)+ely
                self.edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])        
                self.iK = np.kron(self.edofMat,np.ones((8,1))).flatten()
                self.jK = np.kron(self.edofMat,np.ones((1,8))).flatten()           
        # BC's and support
        self.dofs=np.arange(2*(col+1)*(row+1))
        self.fixed=np.union1d(self.dofs[0:2*(row+1):2],np.array([2*(col+1)*(row+1)-1]))
        self.free=np.setdiff1d(self.dofs,self.fixed)
        self.f=np.zeros((self.ndof,1))
        self.u=np.zeros((self.ndof,1))
        self.f[1,0]=-1
        
        
        ### end  constraints
        if height==0:  # todo: extend to 3D
            ##self.grid = np.random.choice(vals, row*col, p=[0.2, 0.8]).reshape(row, col)
            ##self.grid_prop_stress = np.zeros([row,col])
            ##init properties...
            self.grid = np.ones([row, col])
            self.dc = np.ones([row, col])
            self.x=np.ones(col*row,dtype=float)
            self.xold=self.x.copy()
            self.xPhys=self.x.copy()
            self.ce = np.ones(col*row,dtype=float)
            
        return
    
    def bind_agent(self):
        global ke
        L = self.L
        W = self.W
        agents = []

        for i in range(L):
            for j in range(W):
                agt = Agent(ke, self)
                agt.bind_pos([i,j])
                agents.append(agt)
        return agents
        
    def update_state(self):
         #use tools such as FEA to update properties list
        #self.grid_prop_stress = FEA()
        ##def fea(self):
        Emin=1e-9
        Emax=1.0
        penal = 3.0
        
        KE=lk()
        self.H, self.Hs = preFilter(self.W, self.L)
        #print(self.H.size)
        #print(self.Hs.size)
        
        # calculate k and u
        sK=((KE.flatten()[np.newaxis]).T*(Emin+(self.xPhys)**penal*(Emax-Emin))).flatten(order='F')
        K = coo_matrix((sK,(self.iK,self.jK)),shape=(self.ndof,self.ndof)).tocsc()
        K = K[self.free,:][:,self.free]
        self.u[self.free,0]=spsolve(K,self.f[self.free,0])
        #print(self.u.size)
        ##self.stress = self.u[self.free,0] * sK
        
        #ce[:] = (np.dot(self.u[self.edofMat].reshape(self.L*self.W, 8),KE) * self.u[self.edofMat].reshape(self.L*self.W, 8)).sum(1)
        #print(self.u.shape)
        #print(self.edofMat.shape)
        #print(self.L)
        #print(self.W)
        
        # calculate ce
        a1 = self.u[self.edofMat].reshape(self.L*self.W, 8)
        x1 = np.dot(a1,KE) 
        x2 = self.u[self.edofMat].reshape(self.L*self.W, 8)
        self.ce = (x1* x2).sum(1)
        #print(self.ce.size)
        #print(self.xPhys.size)
        
        # calculate obj
        self.obj=( (Emin+self.xPhys**penal*(Emax-Emin))*self.ce ).sum()
        
        b1 = -penal*self.xPhys**(penal-1)*(Emax-Emin)
        cc = np.multiply(b1, self.ce)
        vv = np.ones(self.L*self.W)
        #print(vv.size)
        
        # take ft=1 filter dc and dv
        #self.dc[:] = np.asarray(self.H*(cc[np.newaxis].T/self.Hs))[:,0]   
        #self.dv[:] = np.asarray(self.H*(vv[np.newaxis].T/self.Hs))[:,0]
        cc1 = self.H*(cc[np.newaxis].T/self.Hs)
        self.dc = np.asarray(cc1)[:,0]
        vv1 = self.H*(vv[np.newaxis].T/self.Hs)
        self.dv = np.asarray(vv1)[:,0]
        
        #print(self.dc.size)
        #rint(self.dv.size)
        
          
        
        
        return 

def evolve_by_agents(x,y,ke):
    ## init
    #1. setup up environment(constriants: load, fix, support; properties: density, dc, stress)
    env = Environment()
    env.set_dim(y,x)
    #2. bind agents for environment
    agts = env.bind_agent()
    
    ## evolve
    def agent_update(frame_number, img):   
        nonlocal env
        env.update_state()
        newGrid = env.grid.copy() 
        for a in agts:
            #env.update()
            a.act(newGrid)  # (1)sense (2)decite (3)act
            
        img.set_data(env.grid)           
        return img        
    
    total = 1
    Fact(total=total, state=ON, pos=0)
    Fact(total=total, state=ON, pos=1)
    #ke.add_fact(total=total, state=ON, pos=0)
    
    ke.status()   
    # set up animation
    fig, ax = plt.subplots()
    img = ax.imshow(env.grid, interpolation='nearest')
    ani = animation.FuncAnimation(fig, agent_update, fargs=(img,),
                                  frames = 10,
                                  interval=50,
                                  save_count=50)    
    plt.show() 
    

    


if __name__ == '__main__':
    #parameters:    --x 50   --y 40  --KB r:/rule.txt
    parser = argparse.ArgumentParser(description="A multi agenter toplogy optimizer.")
    parser.add_argument('--x', dest='x', required=True)
    parser.add_argument('--y', dest='y', required=True)
    parser.add_argument('--KB', dest='KB', required=False)      #Knowlege base from txt file. In the first version, we cancle it.
    args = parser.parse_args()
    
    x = 60
    y = 20
    
    if args.x:
        x = int(args.x)
    if args.y:
        y = int(args.y)   
    
    ke = KnowlegeEngine()
    evolve_by_agents(x,y,ke)


 