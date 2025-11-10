# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:41:42 2017

Stock price preidiction based on a mixed deep learning model
with crossentropy, mse and differential reward
correct the code with:
Convolution with windowed dyadic timeseries input, LSTM with n*(n-1)/2 edges input
Mimic the market index with same rise and fall pattern in order to find out the relationship between assets

update on Fri Dec 20 2017
TODO: test the GRU
update the cpu test version to mask the bug
completed pearson and visibility wl kernel methods
tested MSE loss < Cross Entropy loss
updated LSTM network finder
fixed bugs at the hidden status initialing process
tested different optimizer as adam, adagrad, adadelta, rmsprop
tested optimize by SGD with momentum
updated custom regularizer
updated xavier,orthogonal initialization,https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5

update on Fri Dec 20 2017
TODO: test the GRU
completed pearson and visibility wl kernel methods
tested MSE loss < Cross Entropy loss
updated LSTM network finder
fixed bugs at the hidden status initialing process
tested different optimizer as adam, adagrad, adadelta, rmsprop
tested optimize by SGD with momentum
updated custom regularizer
updated xavier,orthogonal initialization,https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5

update on Fri Dec 19 2017
TODO: test the MSE loss and RNN or GRU
tested MSE loss < Cross Entropy loss
updated LSTM network finder
fixed bugs at the hidden status initialing process
tested different optimizer as adam, adagrad, adadelta, rmsprop
tested optimize by SGD with momentum
updated custom regularizer
updated xavier,orthogonal initialization,https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5

update on Fri Dec 18 2017
fixed bugs at the hidden status initialing process
tested different optimizer as adam, adagrad, adadelta, rmsprop
tested optimize by SGD with momentum
updated custom regularizer
TODO: try orthogonal initialization,https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5

update on Fri Dec 13 2017
fixed bugs at the hidden status initialing process
TODO： test different optimizer as adam, adagrad, adadelta, rmsprop
TODO: or optimize by SGD with momentum
TODO: try orthogonal initialization,https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5


update on Fri Dec 10 2017
complete the batch normalization
TODO： test different optimizer as adam, adagrad, adadelta, rmsprop
TODO: or optimize by SGD with momentum

update on Fri Dec 9 2017
TODO: batch normalization
TODO： test different optimizer as adam, adagrad, adadelta, rmsprop
TODO: or optimize by SGD with momentum

update on Fri Dec 8 2017
update content: cuda version

update on Sun Nov 26 2017
update content: preprocess to align all the time series data in the S&P 500
"""
import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn
import networkx as nx
from Models.crnn_factory import CRNN_factory
from data_util import Data_util
import scipy.stats as stats
try:
    from visibility_graph import visibility_graph
except ImportError:
    print("Warning: visibility_graph not available. VWL method will not work.")
    visibility_graph = None
from wlkernel import WLkernerl
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
except ImportError:
    print("Warning: fastdtw not available. DTW method will not work.")
    fastdtw = None
import time
import itertools


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Hyper Parameters
LEARNER_CODE = 'DEEPCNL'  #DEEPCNL, PCC, DTW, VWL
CRNN_CODE = 'CRNN_LSTM' # CRNN_RNN,CRNN_LSTM,CRNN_GRU
DNL_INPLEMENTATION = ['igo','g','io','igof']
WINDOW = 32      # WINDOW size for the time series 32 64
FEATURE_NUM = 4     # feature (e.g. open high, low, close,volume) number, drop open
FILTERS_NUM = 16  # CNN kernel number, LSTM Linear layer merged feature number
LR = 0.0005           # learning rate 0.001 for optimizer
EPOCH_NUM=200       # Iteration times for training data
TICKER_NUM = 12     # S&P500  maximum in data set for 470 tickers for 400 more will be the cudnn bug on batchnorm1d, change to 12
YEAR_SEED = 2 # train_period = 2010+seed-1-1 to 2010+seed-12-31; test_period = 2011+seed-1-1 to 2011+seed-6-30
HIDDEN_UNIT_NUM = 256
HIDDEN_LAYER_NUM = 2
LAM=0.0005
DROPOUT=0.35
DATA_PATH = os.path.join(os.path.dirname(__file__), "Data", "prices-split-adjusted.csv")
SPY_PATH = os.path.join(os.path.dirname(__file__), "Data", "SPY20000101_20171111.csv")
SP500_PATH = os.path.join(os.path.dirname(__file__), "Data", "SP500^GSPC20000101_20171111.csv")

RARE_RATIO = 0.002 # 0.001 for 470 / 0.01 for less
TOP_DEGREE_NODE_NUM=20

# 50
# https://www.guggenheiminvestments.com/etf/fund/xlg-guggenheim-sp-500-top-50-etf/holdings
OEX = ['AAPL','MSFT','AMZN','FB','BRKB','JNJ','JPM','XOM','GOOG','GOOGL','BAC','WFC','CVX','HD','PG','UNH','T','PFE','V','VZ','INTC','C','CSCO','BA','CMCSA','KO','DWDP','MRK','PEP','DIS','ABBV','PM','ORCL','MA','GE','WMT','MMM','IBM','MCD','AMGN','MO','HON','TXN','MDT','UNP','SLB','GILD','ABT','BMY','QCOM','CAT','UTX','ACN','PCLN','PYPL','UPS','GS','USB','SBUX','LOW','COST','NKE','LMT','LLY','CVS','CELG','MS','BIIB','AXP','TWX','COP','NEE','BLK','CHTR','CL','FDX','WBA','MDLZ','DHR','BK','AGN','OXY','GD','RTN','GM','MET','AIG','DUK','MON','SPG','COF','KHC','F','EMR','HAL','SO','TGT','FOXA','KMI','EXC','ALL','BLKFDS','FOX','USD','MSFUT','ESH8']

# 100
# https://www.ishares.com/us/products/239723/ishares-sp-100-etf
XLG = ['AAPL','MSFT','AMZN','FB','BRK','B','JNJ','JPM','XOM','GOOG','GOOGL','BAC','WFC','CVX','HD','PG','UNH','T','PFE','V','VZ','INTC','C','CSCO','BA','CMCSA','KO','MRK','PEP','DIS','ABBV','PM','ORCL','MA','GE','WMT','MMM','IBM','MCD','AMGN','MO','HON','MDT','UNP','AVGO','SLB','GILD','BMY','UTX','PCLN','SBUX','CELG','X9USDFUTC']

# 200
# https://www.ishares.com/us/products/239721/ishares-russell-top-200-etf
IWL = ['AAPL','MSFT','AMZN','FB','BRKB','JNJ','JPM','XOM','GOOG','GOOGL','BAC','WFC','CVX','HD','PG','UNH','T','V','PFE','VZ','INTC','C','CSCO','BA','CMCSA','KO','DWDP','DIS','PEP','MRK','ABBV','PM','MA','GE','WMT','ORCL','MMM','IBM','MCD','AMGN','MO','NVDA','HON','TXN','MDT','UNP','AVGO','SLB','GILD','BMY','QCOM','UTX','ABT','ACN','ADBE','CAT','PCLN','PYPL','UPS','GS','NFLX','USB','SBUX','LOW','TMO','LLY','COST','LMT','NKE','CVS','CELG','PNC','CRM','BIIB','AXP','COP','BLK','MS','TWX','NEE','CB','FDX','WBA','SCHW','CHTR','CL','EOG','MDLZ','ANTM','AMAT','DHR','BDX','AGN','AET','OXY','AMT','BK','RTN','GM','AIG','DUK','ADP','SYK','GD','DE','PRU','CI','MON','ITW','SPG','ATVI','CME','NOC','COF','TJX','CSX','MU','D','ISRG','KHC','MET','F','EMR','PX','PSX','TSLA','ESRX','HAL','SPGI','SO','NSC','CTSH','ICE','MAR','VLO','CCI','TGT','BBT','MMC','KMB','NXPI','INTU','HPQ','DAL','STT','HUM','VRTX','FOXA','WM','ALL','LYB','TRV','KMI','EXC','ETN','EBAY','BSX','JCI','MCK','LUV','APD','STZ','ECL','SHW','EQIX','AFL','AON','BAX','GIS','EA','AEP','APC','PXD','SYY','GLW','REGN','PPG','PSA','EL','CCL','YUM','MNST','HPE','ALXN','BLKFDS','LVS','HCA','KR','ADM','PCG','EQR','CBS','TMUS','USD','FOX','BEN','DISH','VMW','BHF','S','JPFFT','ESH8']

def check_required_data():
    """Check if all required data files exist"""
    data_dir = os.path.join(os.path.dirname(__file__), "Data")
    
    required_files = {
        'prices-split-adjusted.csv': True,  # Must exist
        'SPY20000101_20171111.csv': False,   # Can download
        'SP500^GSPC20000101_20171111.csv': False  # Can download
    }
    
    missing_critical = []
    missing_downloadable = []
    
    for filename, is_critical in required_files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            if is_critical:
                missing_critical.append(filename)
            else:
                missing_downloadable.append(filename)
    
    if missing_critical:
        print("\n" + "="*70)
        print("❌ CRITICAL DATA FILES MISSING")
        print("="*70)
        for f in missing_critical:
            print(f"  ❌ {f}")
        print("\nPlease add these files to the Data/ folder before running.")
        print("="*70)
        return False
    
    if missing_downloadable:
        print("\n" + "="*70)
        print("⚠️  MARKET DATA FILES MISSING")
        print("="*70)
        for f in missing_downloadable:
            print(f"  ❌ {f}")
        print("\nRun this command to download missing data:")
        print("  python download_market_data.py")
        print("="*70)
        
        response = input("\nTry to download now? (y/n): ")
        if response.lower() == 'y':
            try:
                import yfinance as yf
                os.makedirs(data_dir, exist_ok=True)
                
                if 'SPY20000101_20171111.csv' in missing_downloadable:
                    print("\n📥 Downloading SPY data...")
                    spy = yf.download("SPY", start="2000-01-01", end="2017-11-12")
                    spy.to_csv(os.path.join(data_dir, 'SPY20000101_20171111.csv'))
                    print("✅ SPY downloaded")
                
                if 'SP500^GSPC20000101_20171111.csv' in missing_downloadable:
                    print("\n📥 Downloading S&P 500 Index data...")
                    sp500 = yf.download("^GSPC", start="2000-01-01", end="2017-11-12")
                    sp500.to_csv(os.path.join(data_dir, 'SP500^GSPC20000101_20171111.csv'))
                    print("✅ S&P 500 downloaded")
                
                print("\n✅ All data files ready!")
                return True
                
            except Exception as e:
                print(f"\n❌ Auto-download failed: {e}")
                print("Please run: python download_market_data.py")
                return False
        else:
            return False
    
    print("✅ All required data files present")
    return True


class Experimental_platform:
    def __init__(self, datatool):
        self.model = None
        self.datatool = datatool
        self.wlkernel = WLkernerl()
        self.device = DEVICE
        # Don't initialize crnn_factory here - do it after we know actual ticker count

    def regularizer(self, lam, loss):
        """L2 regularization on model parameters"""
        l2_reg = torch.tensor(0., device=self.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param)
        return loss + lam * l2_reg

    def top_degree_nodes(self, g, top_n=TOP_DEGREE_NODE_NUM):
        """Print top N nodes by degree"""
        if len(g.nodes()) == 0:
            print("Graph has no nodes")
            return
        
        sorted_degrees = sorted(g.degree, key=lambda x: x[1], reverse=True)
        print(f"\n[Top {min(top_n, len(sorted_degrees))} nodes by degree]")
        for i, (node, degree) in enumerate(sorted_degrees[:top_n]):
            print(f"{i+1}. {node}: degree={degree}")

    def train_model(self, x, y):
        """Train model with data - FIXED to use actual ticker count"""
        # CRITICAL FIX: Initialize model with actual ticker count from data
        actual_ticker_num = self.datatool.actual_ticker_num
        
        print(f"🔧 Initializing model with {actual_ticker_num} tickers (originally requested {TICKER_NUM})")
        
        # Create factory with actual ticker count
        crnn_factory = CRNN_factory(
            FEATURE_NUM, 
            FILTERS_NUM, 
            WINDOW, 
            actual_ticker_num,  # Use actual count, not TICKER_NUM
            HIDDEN_UNIT_NUM, 
            HIDDEN_LAYER_NUM, 
            DROPOUT
        )
        
        self.model = crnn_factory.get_model(CRNN_CODE)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.loss_func = torch.nn.CrossEntropyLoss()

        print('🚀 Starting model training...')
        if self.device.type == 'cuda':
            cudnn.benchmark = True
        
        # Check if we have enough data
        if len(y) == 0:
            raise ValueError("No target labels available! Training data is too small for the window size.")
        
        inputs = Variable(torch.from_numpy(x)).float().to(self.device)
        targets = Variable(torch.from_numpy(y)).long().to(self.device)
        
        print(f"📊 Training data shape: {inputs.shape}")
        print(f"📊 Target labels: {len(targets)}")
        
        self.model.train()
        prediction = None
        
        for epoch in range(EPOCH_NUM):
            self.model.zero_grad()
            self.model.hidden = self.model.init_hidden()

            prediction = self.model(inputs)
            loss = self.loss_func.forward(self.model.classify_result(prediction), targets)
            loss = self.regularizer(LAM, loss)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f'  Epoch {epoch:3d}: loss = {loss.data.cpu().numpy():.6f}')
                
        result = self.model.classify_result(prediction)
        torch.save(self.model, 'model.pt')
        print('✅ Model training completed and saved')
        return loss.data.cpu().numpy()

    def test_model(self, x, y):
        """Test model - FIXED to handle device and actual ticker count"""
        self.model = torch.load('model.pt', map_location=self.device)
        self.model.eval()
        self.loss_func = self.loss_func.to(self.device)
        print('🧪 Testing the learned model...')
        
        inputs = Variable(torch.from_numpy(x)).to(self.device)
        targets = Variable(torch.from_numpy(y)).to(self.device)

        prediction = self.model(inputs.float())
        result = self.model.classify_result(prediction)
        loss = self.loss_func.forward(result, targets.long())
        print(f'  Test loss: {loss.cpu().data.numpy():.6f}')
        
        pred = result.max(1, keepdim=True)[1]
        correct = pred.eq(targets.view_as(pred).long()).sum()
        acc = (correct.float() / len(targets)).cpu().data.numpy()
        print(f'  Accuracy: {acc:.4f} ({100*acc:.2f}%)')
        return loss.cpu().data.numpy(), acc

    def DNL_graph_learning(self, dnl_implementation, rare_ratio):
        W = None
        for m in self.model.modules():
            if isinstance(m, nn.LSTM):
                (W_ii, W_if, W_ig, W_io) = m.weight_ih_l0.view(4, HIDDEN_UNIT_NUM, -1)
                if dnl_implementation == 'igo':
                    W = W_ii + W_ig + W_io
                if dnl_implementation == 'igof':
                    W = W_ii + W_io + W_ig + W_if
                if dnl_implementation == 'io':
                    W = W_ii + W_io
                if dnl_implementation == 'g':
                    W = W_ig
    
            if isinstance(m, nn.RNN):
                W = m.weight_ih_l0
            if isinstance(m, nn.GRU):
                (W_ir, W_iz, W_in) = m.weight_ih_l0.view(3, HIDDEN_UNIT_NUM, -1)
                W = W_iz + W_in + W_ir
                
        W = W.cumsum(dim=0)[HIDDEN_UNIT_NUM - 1]
        W = W.sort(descending=True)
        E = W[1]  # ticker dyadics
        W = W[0]  # ticker weights
        g = nx.Graph()
        edge_bunch = []
        
        # FIXED: Use actual_ticker_num from datatool
        actual_num = self.datatool.actual_ticker_num
        for k in range(0, int(actual_num * (actual_num - 1) * 0.5 * rare_ratio)):
            if W[k].cpu().data.numpy() > 0:
                (i, j) = self.datatool.check_dyadic(E[k].cpu().data.numpy())
                i = self.datatool.check_ticker(i)
                j = self.datatool.check_ticker(j)
                edge_bunch.append((i, j, W[k].cpu().data.numpy()))
                
        g.add_weighted_edges_from(edge_bunch)
        self.top_degree_nodes(g)
        return g


    def deep_CNL(self, dnl_implementation, train_x, train_y, rare_ratio):
        print('[DeepCNL]')
        self.train_model(train_x, train_y)
        return self.DNL_graph_learning(dnl_implementation, rare_ratio)

    def Pearson_cor(self,rare_ratio):
        print('[Pearson correlation coefficients on time series tuples with (coefficient, p-value)]')
        g = nx.Graph()
        edgelist = {}
        for n in range(0,self.datatool.compare_data.shape[0]):
            t=self.datatool.compare_data[n]
            (i, j) = self.datatool.check_dyadic(n)
            i = self.datatool.check_ticker(i)
            j = self.datatool.check_ticker(j)
            prs=stats.pearsonr(t[0],t[1])
            if prs[1] <= 0.01 and prs[0]>0: # prs[1]: p value = edge_ratio; prs[0] = coefficient
                edgelist[(i,j)]=prs[0]
        edgelist=sorted(edgelist.items(),key=lambda x:x[1],reverse=True)
        edge_bunch = []
        for k in range(0, int(TICKER_NUM * (TICKER_NUM - 1) * 0.5 * rare_ratio)):
            ((i, j), weight) = edgelist[k]
            # g.add_edge(i, j)
            edge_bunch.append((i, j, weight))
        g.add_weighted_edges_from(edge_bunch)
        self.top_degree_nodes(g)
        return g


    def VWL_graph(self,rare_ratio):
        print('[Visibility graphs-WL kernel method]')
        g=nx.Graph()
        edgelist={}
        for n in range(0,self.datatool.compare_data.shape[0]):
            t=self.datatool.compare_data[n]
            (i, j) = self.datatool.check_dyadic(n)
            i = self.datatool.check_ticker(i)
            j = self.datatool.check_ticker(j)
            g0=visibility_graph(t[0])
            g1=visibility_graph(t[1])
            weight=self.wlkernel.compare(g0,g1,h=10,node_label=False)
            print(i,j,weight)
            edgelist[(i, j)] = weight
        edgelist = sorted(edgelist.items(), key=lambda x: x[1], reverse=True)
        edge_bunch = []
        for k in range(0, int(TICKER_NUM * (TICKER_NUM - 1) * 0.5 * rare_ratio)):
            ((i, j), weight) = edgelist[k]
            # g.add_edge(i, j)
            edge_bunch.append((i, j, weight))
        g.add_weighted_edges_from(edge_bunch)
        self.top_degree_nodes(g, TOP_DEGREE_NODE_NUM)
        return g

    def DTW_graph(self,rare_ratio):
        print('[DTW graph finding]')
        g = nx.Graph()
        edgelist={}
        for n in range(0, self.datatool.compare_data.shape[0]):
            t = self.datatool.compare_data[n]
            (i, j) = self.datatool.check_dyadic(n)
            i = self.datatool.check_ticker(i)
            j = self.datatool.check_ticker(j)
            distance, path = fastdtw(t[0], t[1], dist=euclidean)
            if n%100==0:
                print(i,j,distance)
            edgelist[(i, j)] = 1.0/(distance+1)
        edgelist = sorted(edgelist.items(), key=lambda x: x[1], reverse=True)
        edge_bunch=[]
        for k in range(0,int(TICKER_NUM*(TICKER_NUM-1)*0.5*rare_ratio)):
            ((i,j),weight)=edgelist[k]
            #g.add_edge(i, j)
            edge_bunch.append((i,j,weight))
        g.add_weighted_edges_from(edge_bunch)
        self.top_degree_nodes(g, TOP_DEGREE_NODE_NUM)
        return g


    '''
    Experiments
    '''

    def coverage_comparison(self):
        benchmark=[['NFLX','FFIV','CMI','AIG','ZION','HBAN','AKAM','PCLN','WFMI','Q'], # 2010
                   ['COG','EP','ISRG','MA','BIIB','HUM','CMG','PRGO','OKS','ROST'], # 2011
                   ['HW','DDD','REGN','LL','PHM','MHO','AHS','VAC','S','EXH'], # 2012
                   ['NFLX','MU','BBY','DAL','CELG','BSX','GILD','YHOO','HPQ','LNC'], # 2013
                   ['LUV','EA','EW','AGN','MNK','AVGO','GMCR','DAL','RCL','MNST'], # 2014
                   ['NFLX','AMZN','ATVI','NVDA','CVC','HRL','VRSN','RAI','SBUX','FSLR'], # 2015
                   ['NVDA','OKE','FCX','CSC','AMAT','PWR','NEM','SE','BBY','CMI']] # 2016
        
        for seed in [YEAR_SEED]:
            train_period, test_period = self.period_generator(seed)
            
            print(f"\n{'='*60}")
            print(f"Year: {2010 + seed}")
            print(f"Training period: {train_period[0]} to {train_period[1]}")
            print(f"Testing period: {test_period[0]} to {test_period[1]}")
            print(f"{'='*60}\n")
            
            try:
                # Load data
                train_x = self.datatool.load_x(train_period)
                train_y = self.datatool.load_y(train_period)
                
                # CRITICAL VALIDATION: Check if we have enough training data
                min_required_labels = 10  # Need at least 10 samples for meaningful training
                
                if len(train_y) < min_required_labels:
                    print(f"\n⚠️  ERROR: Insufficient training data!")
                    print(f"   Available labels: {len(train_y)}")
                    print(f"   Required: At least {min_required_labels}")
                    print(f"\n   Possible solutions:")
                    print(f"   1. Use longer training period (currently: {train_period})")
                    print(f"   2. Reduce WINDOW size (currently: {WINDOW})")
                    print(f"   3. Use different year (YEAR_SEED={seed})")
                    print(f"\n   Calculation:")
                    print(f"   - Need: (trading_days - WINDOW) > {min_required_labels}")
                    print(f"   - With WINDOW={WINDOW}: need >{WINDOW + min_required_labels} trading days")
                    continue
                
                # Proceed with training
                if LEARNER_CODE == 'DEEPCNL':
                    g = self.deep_CNL('igo', train_x, train_y, RARE_RATIO)
                elif LEARNER_CODE == 'PCC':
                    g = self.Pearson_cor(RARE_RATIO)
                else:
                    print(f"Unknown LEARNER_CODE: {LEARNER_CODE}")
                    continue
                
                # Calculate coverage
                sum_count = 0.
                covered_stocks = []
                for ticker in benchmark[seed]:
                    if ticker in g.nodes() and g.degree(ticker) > 0:
                        print(f'[COVERED] {ticker}')
                        covered_stocks.append(ticker)
                        sum_count += 1
                    else:
                        print(f'[MISSED]  {ticker}')
                
                print(f'\n📊 Results for {2010 + seed}:')
                print(f'   Covered: {sum_count}/{len(benchmark[seed])} stocks')
                print(f'   Hit ratio: {sum_count/len(benchmark[seed])*100:.1f}%')
                print(f'   Covered stocks: {", ".join(covered_stocks)}')
                
            except ValueError as e:
                print(f"\n❌ Error processing year {2010 + seed}: {e}")
                continue
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                continue
            
    def get_rank(self,sorted_degree_list,ticker):
        for i in range(0,len(sorted_degree_list)):
            if ticker == sorted_degree_list[i][0]:
                return i



    def rise_fall_prediction(self, seed):
        train_period, test_period = self.period_generator(seed)
        print('[RISE-FALL PREDICT TASK]')
        print('train with data in', train_period)
        print('test with data in', test_period)
        train_x = self.datatool.load_x(train_period)
        train_y = self.datatool.load_y(train_period)
        test_x = self.datatool.load_x(test_period)
        test_y = self.datatool.load_y(test_period)
        train_loss = 0.
        test_loss = 0.
        accuracy = 0.
        print('[REPEAT Experiments, iteration]',seed)
        train_loss=self.train_model(train_x,train_y)
        loss,acc=self.test_model(test_x,test_y)
        test_loss=loss
        accuracy=acc
        print('[**RESULT**]')
        print('Train loss',train_loss)
        print('Test loss', test_loss)
        print('accuracy', accuracy)

    def DNL_density_comparison(self,seed):
        print('DNL implemented with gates:')
        train_period,test_period = self.period_generator(seed)
        print('train with data in',train_period)
        # load data
        train_x = self.datatool.load_x(train_period) #TRAIN_PERIOD
        train_y = self.datatool.load_y(train_period) # TRAIN_PERIOD
        self.train_model(train_x, train_y)
        for dnl in DNL_INPLEMENTATION:
            print('DeepCNL-'+dnl)
            g=self.DNL_graph_learning(dnl,RARE_RATIO)
            print('edge density')
            print('[XLG 50]', experiment.edge_density(g, XLG))
            print('[OEX 100]', experiment.edge_density(g, OEX))
            print('[IWL 200]', experiment.edge_density(g, IWL))

    def ALL_density_comparison(self, seed):
        print('general comparison with DeepCNL-igo, PCC')
        g = experiment.influential_asset_finding(seed)
        print('average weight')
        print('edge density')
        print('[XLG 50]', experiment.edge_density(g, XLG))
        print('[OEX 100]', experiment.edge_density(g, OEX))
        print('[IWL 200]', experiment.edge_density(g, IWL))

    def correlation_degree_comparison(self, seed):
        g = experiment.influential_asset_finding(seed)
        print('average weight')
        print('[XLG]', experiment.average_weight(nx.subgraph(g, XLG)))
        print('[OEX]', experiment.average_weight(nx.subgraph(g, OEX)))
        print('[IWL]', experiment.average_weight(nx.subgraph(g, IWL)))
        print('[SPY]', experiment.average_weight(g))

    '''
    seed from 0 to 6
    '''
    def period_generator(self,seed):
        # train_period = [str(2010 + seed) + '-1-1', str(2010 + seed) + '-12-31']
        # test_period = [str(2011 + seed) + '-1-1', str(2011 + seed) + '-6-30']
        # return train_period,test_period
        train_period = [str(2010 + seed) + '-01-01', str(2010 + seed) + '-12-31']
        test_period = [str(2011 + seed) + '-01-01', str(2011 + seed) + '-06-30']
        return train_period, test_period

    def influential_asset_finding(self,seed):
        print('DNL implemented with gates:')
        train_period,test_period = self.period_generator(seed)
        print('train with data in',train_period)
        # load data
        train_x = self.datatool.load_x(train_period) #TRAIN_PERIOD
        train_y = self.datatool.load_y(train_period) # TRAIN_PERIOD

        if LEARNER_CODE == 'DEEPCNL':
            g = self.deep_CNL("igo", train_x,train_y,RARE_RATIO)
        if LEARNER_CODE == 'PCC':
            g = self.Pearson_cor(RARE_RATIO)
        if LEARNER_CODE == 'VWL':
            g = self.VWL_graph(RARE_RATIO)
        if LEARNER_CODE == 'DTW':
            g = self.DTW_graph(RARE_RATIO)

        #nx.draw(g, nx.spring_layout(g), with_labels=True, font_size=20)
        #plt.show()
        return g

    def average_weight(self,g):
        ws = nx.get_edge_attributes(g, 'weight')
        result=0.
        for e in ws:
            result+=ws[e]
        return result/len(ws)


    def edge_density(self,g,etf):
        c=0.
        combines=[c for c in itertools.combinations(range(len(etf)),2)]
        for (i,j) in combines:
            if g.has_edge(etf[i],etf[j]):
                c+=1
        return c/len(combines)







if __name__ == "__main__":
    if not check_required_data():
        print("\n❌ Cannot proceed without required data files. Exiting...")
        exit(1)

    print('[Parameters]***********************************************')
    print('LEANER_CODE',LEARNER_CODE)
    print('CRNN_CODE', CRNN_CODE)
    print('DNL_IMPLEMENTATIONS', DNL_INPLEMENTATION)
    print('[WINDOW]',WINDOW)
    print('[FEATURE_NUM]',FEATURE_NUM)
    print('[FILTERS_NUM]',FILTERS_NUM)
    print('[LR]',LR)
    print('[DeepCNL epoches]',EPOCH_NUM)
    print('[TICKER_NUM]',TICKER_NUM)
    print('[HIDDEN_UNIT_NUM]',HIDDEN_UNIT_NUM)
    print('[HIDDEN_LAYER_NUM]',HIDDEN_LAYER_NUM)
    print('[LAM]',LAM)
    print('[DROPOUT]',DROPOUT)
    print('[RARE_RATIO]',RARE_RATIO)
    print('[TOP_DEGREE_NODE_NUM]',TOP_DEGREE_NODE_NUM)
    print('[Parameters]***********************************************')
    datatool=Data_util(TICKER_NUM,WINDOW,FEATURE_NUM,DATA_PATH,SPY_PATH)
    experiment=Experimental_platform(datatool)
    start = time.time()
    # experiment.coverage_comparison()
    #for seed in [5,6]:
    #experiment.DNL_density_comparison(0)
        #experiment.ALL_density_comparison(seed)
        #   experiment.rise_fall_prediction(seed)

    # experiment.correlation_degree_comparison(YEAR_SEED)
    #experiment.sum_degree()


    elapsed = (time.time() - start)
    print("Time used:", elapsed, 'Seconds')
    
    
   
    
