#from Sapsan import Sapsan
import matplotlib as mpl
#mpl.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow

class Results:

    def __init__(self):

        params = {'axes.labelsize': 20, 'legend.fontsize': 15, 'xtick.labelsize': 17,'ytick.labelsize': 17,
              #'axes.tick_params':21, 
              'axes.titlesize':20, 'axes.linewidth': 1, 'lines.linewidth': 1.5,
              'xtick.major.width': 1,'ytick.major.width': 1,'xtick.minor.width': 1,'ytick.minor.width': 1,
              'xtick.major.size': 4,'ytick.major.size': 4,'xtick.minor.size': 3,'ytick.minor.size': 3,
              'axes.formatter.limits' : [-7, 7], 'text.usetex': False}#,  'figure.figsize': [9,6]}
        mpl.rcParams.update(params)
        
    def defaults(self, var, kwargs):
        default = {'vmin':min(var.flatten()),
           'vmax':max(var.flatten()),
           'cmap':'viridis',
           'name':''}

        for key in default:
            if key not in kwargs:
                kwargs[key]=default[key]
        return kwargs
        
    def prediction(self, var, pred,block, block_size, **kwargs):  
        #reformats prediction; topology comparison plots
        kwargs = self.defaults(var, kwargs)
        
        print('---Prediction---')
        print('var, pred in', np.shape(var), np.shape(pred), self.dim, self.ttrain)
        
        if self.axis==2: pred = np.reshape(pred, (self.dim,self.dim)); var = np.reshape(var, (self.dim,self.dim)) 
        if self.axis==3: 
            print('3d plotting prediction')
            pred = np.reshape(pred, (block[0],block[1],block[2],block_size[0],block_size[1],block_size[2]))
            pred = pred.transpose(0,3,1,4,2,5)
            #print('ERROR:', pred.shape)int(self.dim*len(self.ttrain)*self.train_fraction),self.dim,self.dim)
            pred = pred.reshape(int(self.dim*self.train_fraction),self.dim,self.dim)
            var = np.reshape(var, (block[0],block[1],block[2],block_size[0],block_size[1],block_size[2]))
            var = var.transpose(0,3,1,4,2,5).reshape(int(self.dim*self.train_fraction),self.dim,self.dim)
        print('var, pred out', np.shape(var), np.shape(pred))
        fig = plt.figure(figsize = (16, 6))
        a = fig.add_subplot(121)
        im = plt.imshow(var[0], cmap=kwargs['cmap'], vmin=kwargs['vmin'], vmax = kwargs['vmax'])
        plt.colorbar(im).ax.tick_params(labelsize=14)
        plt.title('Original %s'%kwargs['name'])
        #if self.savepath:
        #    fig.tight_layout()
        #    fig.savefig(self.savepath+'orig.png')
        
        
        #fig = plt.figure(figsize = (8, 6))
        a = fig.add_subplot(122)
        im = plt.imshow(pred[0], cmap=kwargs['cmap'], vmin=kwargs['vmin'], vmax = kwargs['vmax'])
        plt.colorbar(im).ax.tick_params(labelsize=14)
        plt.title('Predicted %s'%kwargs['name'])
        if self.savepath:
            fig.tight_layout()
            fig.savefig(self.savepath+'pred.png')
            mlflow.log_artifact(self.savepath+'pred.png')

        return pred
    
    
    def pdf(self, var, bins=100):
        #produces a pdf plot
        print('Plotting the PDF')
        
        if len(var)==2: names = ['original', 'predicted']
        else: 
            names = range(len(var))
            names = [str(i) for i in names]
        
        plt.figure('pdf', figsize = (6, 6))

        if len(np.shape(var))>1:
            func = []
            
            #>>>>FIX UP!
            #val = np.zeros((np.shape(var)[0],self.dim**self.axis))    
            val = np.zeros((np.shape(var)))    
            for i in range(len(var)):
                val[i] = np.sort(var[i])
                count = 0
                for b in range(len(val[i])):
                    if val[i][b] > 0.012:
                        count+=1
                print('a number of outliers', count)
                
                plt.hist(val[i], bins=bins, lw=3, normed=True, histtype='step', label=names[i])
                
        else:
            plt.hist(var.flatten(), bins=100, lw=3, normed=True, histtype='step')
        plt.yscale('log')
        plt.legend()
        plt.xlabel('data values')
        plt.ylabel('PDF')
        if self.savepath:
            plt.tight_layout()
            plt.savefig(self.savepath+'pdf.png')
            mlflow.log_artifact(self.savepath+'pdf.png')

        return

        
    def cdf(self, var):
        from scipy.stats import ks_2samp
        from scipy.interpolate import interp1d

        #produces a cdf plot
        print('Plotting the CDF')
        if len(var)==2: names = ['original', 'predicted']
        else: 
            names = range(len(var))
            names = [str(i) for i in names]
        
        plt.figure('cdf', figsize=(6,6))
        func = []
        #val = np.zeros((np.shape(var)[0],self.dim**self.axis))
        val = np.zeros((np.shape(var)))
        for i in range(len(var)):
            val[i] = np.sort(var[i])

            #cdf calculation via linear interpolation
            length = len(val[i])
            yvals = np.linspace(0,length-1, length)/length
            plt.plot(val[i], yvals, label=names[i])
            func.append(interp1d(val[i], yvals))                                                            

            if i==1:
                ks_stat, pvalue = ks_2samp(val[0], val[1])
                minima = max([min(val[0]), min(val[1])])
                maxima = min([max(val[0]), max(val[1])])

                xtest = np.linspace(minima, maxima, length*10)
                D = abs(func[0](xtest)-func[1](xtest))
                Dmax = max(D)
                Dpos = xtest[np.argmax(D)]
                plt.axvline(x=Dpos, linewidth=1, color='tab:red', linestyle='--')

                txt = ('pvalue = %.3e\n'%pvalue+
		     r'$\rm ks_{stat}$'+' = %.3e\n'%ks_stat+
		     r'$\rm ks_{line}$'+' = %.3e\n'%Dmax+
		     r'$\rm line_{pos}$'+' = %.3e'%Dpos)
                
                plt.figtext(0.17, 0.6, txt, fontsize=14)        
        
        plt.legend()
        plt.xlabel('data values')
        plt.ylabel('CDF')
        plt.title('t = %.4f'%self.ttest)
        if self.savepath:
            plt.tight_layout()
            plt.savefig(self.savepath+'cdf.png')            
            mlflow.log_artifact(self.savepath+'cdf.png')
        plt.show()
        plt.clf()
        mlflow.log_metric('ks_stat', ks_stat)
        

        try:
            return pvalue, ks_stat, Dmax, Dpos
        except:
            return 100, 100, 100, 100
        
        
    def parity(self, var, pred):
        from scipy.stats import kde

        #parity plots
        x = var.flatten()
        y = pred.flatten()
        vmin = min(var.flatten())*1.1
        vmax = max(var.flatten())*1.1
        vymin = min(pred.flatten())*1.1
        vymax = max(pred.flatten())*1.1
         
        # Evaluate a gaussian kde on a regular grid of nbins x over data extents
        nbins=300
        k = kde.gaussian_kde([x,y])
        xi, yi = np.mgrid[vmin:vmax:nbins*1j, vymin:vymax:nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        
        plt.figure(figsize=(7,6))
        zi = np.ma.masked_where(zi <= 1, zi)
        cmap = plt.cm.get_cmap('Oranges')

        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap=cmap,norm=colors.LogNorm())
        plt.colorbar().ax.tick_params(labelsize=14)
        one = np.linspace(vmin, vmax)
        plt.plot(one,one,'--', c='tab:red', label='y=x', alpha=0.5)
        plt.xlim([vmin,vmax])
        plt.ylim([vymin,vymax])
        if self.savepath:
            plt.tight_layout()
            plt.savefig(self.savepath+'parity.png')
            mlflow.log_artifact(self.savepath+'parity.png')
            

    def hyper3d(self, model, quantity):
        from mpl_toolkits.mplot3d import Axes3D

        print('best parameters', model.best_estimator_)
        print(model.cv_results_[quantity])
        for keys,values in model.cv_results_.items(): print(keys)

        length = len(model.cv_results_['params'])
        alpha = np.zeros(length)
        gamma = np.zeros(length)
        result = np.zeros(length)
        
        for i in range(length):
            alpha[i] = model.cv_results_['param_alpha'][i]
            gamma[i] = model.cv_results_['param_gamma'][i]
            result[i] = model.cv_results_[quantity][i]
            
            if alpha[i] == model.best_params_['alpha'] and gamma[i] == model.best_params_['gamma']:
                besta = np.log10(alpha[i])
                bestg = np.log10(gamma[i])
                bestr = result[i]
        
        alpha = [np.log10(i) for i in alpha]
        gamma = [np.log10(i) for i in gamma]
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(alpha, gamma, result, marker = 'o', label='search')
        ax.scatter(besta, bestg, bestr, marker = 'o', s=30, color='r', label='best')

        if len(alpha)>1:
            ax.plot_trisurf(alpha, gamma, result, alpha = 0.5)
            ax.set_xlabel('log(alpha)')
            ax.set_ylabel('log(gamma)')
            ax.set_zlabel(quantity)
        if self.savepath:
            fig.tight_layout()
            fig.savefig(self.savepath+'3Dpars.png')
            

    def plot_pars(self, ax, scores, x, index, yname, xname, ticks):
        plt.plot(x, scores[:,index], 'o-', label='train')
        plt.plot(x, scores[:,index+1], 'o-', label='test')
        if ticks==True: 
            plt.gca().set_xticks(x)
        else:
            ax.get_xaxis().set_major_locator(MaxNLocator(integer=True)) 
        plt.legend()
        plt.grid(True)
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.title(r'$\tau_{1%d}$'%self.targetComp)
        #if 'Mean' in yname:
        #    plt.ylim([0.000-0.001*0.2,0.0055])
        #else:
        #    plt.ylim([0.35,1.02])
    
    
    def plot_scores(self, x=None, names=None, xtitle='Setup', ticks=False, size=[12,4]):
        if self.savepath==None: sys.exit("Path is not indicated: set 'functions.savepath' to a path")
        scores = np.genfromtxt(self.savepath+'scores.dat', skip_header=1)

        if len(np.shape(scores))==1: scores=np.array([scores])
        scores = scores[:,1:]
        
        length=len(scores)

        if x!=None:pass 
        else: x = np.linspace(1, length, length)
        
        
        fig = plt.figure(figsize = (size[0], size[1]))
        ax = fig.add_subplot(1,2,1)
        self.plot_pars(ax, scores, x, 0, 'Mean_Absolute_Error', xtitle, ticks)
        ax = fig.add_subplot(1,2,2)
        self.plot_pars(ax, scores, x, 2, 'Explained_Variance_Score', xtitle, ticks)

        if names!=None:
            setups = ''
            for i in range(length): setups=setups+'[%s] '%(i)+names[i]+'\n'
            #for i in range(length): setups+'[%s]'%(i)        
            plt.figtext(0.4, 0, setups, fontsize=15, horizontalalignment='left', verticalalignment='top')
        plt.tight_layout()
        plt.savefig(self.savepath+'scores.png')
        

    def Check_Directories(self):
        #create appropriate directories if those don't exist
        
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        else:
            pass
        return

    def output(self, vals, target, out_format=None):
        import xlsxwriter
        '''
        if self.path==None:
            inpath = '%s.h5'%var  
        elif '@' in self.path:
            i = self.path.find('@')
            try: 
                n = int(self.path[i+1])
                inpath = self.path.replace(self.path[i:i+2], '%0*d' % (n, self.ttrain[0]*self.dt))
                
            except:   
                n = int(self.path[i+2])
                inpath = self.path.replace(self.path[i:i+3], '%.*f' % (n, self.ttrain[0]*self.dt))
                
        else: inpath = self.path
        '''
        vals = np.concatenate((vals, target), axis=1)

        savename = self.savepath+'output'#'%s%dd_dim%d_t%.4f'%(inpath, self.axis, self.dim, self.ttrain[0]*self.dt)
        
        #>>>Fix pars to be read automatically from parameters<<<
        pars=['u0', 'u1','u2',
              'b0','b1','b2',
              'a0','a1','a2',
              'du00', 'du01','du02','du10','du11','du12', 'du20','du21','du22', 
              'db00', 'db01','db02','db10','db11','db12', 'db20','db21','db22',
              'da00', 'da01','da02','da10','da11','da12', 'da20','da21','da22','tn0','tn1','tn2']
        
        if out_format=='xlsx':
            workbook = xlsxwriter.Workbook(savename+'.xlsx')
            worksheet = workbook.add_worksheet()

            print(vals[0][:])
            print('par shape', np.shape(pars))
            print(np.shape(vals))
            for row in range(np.shape(vals)[0]+1):
                for col in range(np.shape(vals)[-1]):
                    #print(col, pars[col])
                    if row==0: worksheet.write(row,col,pars[col])
                    else: worksheet.write(row, col, vals[row-1][col])

            workbook.close()
        else:
            np.savetxt(savename+'.txt', vals, header="\t".join(pars))

        
        mlflow.log_artifact(savename+'.txt')
    
    #careful with boundaries when selecting train and test in regards to gradient

