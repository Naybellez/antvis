from modelManagment import get_lin_lay
class Cards:
    def __init__(self):
        model_card_vgg = {'name': 'vgg', 'model': 'vgg16',
                          'f_lin_lay':[200704,    # 452 144
                                     200704,      # 226 72
                                     200704,#14336,       # 113 36
                                     200704,#3584,        # 57 18 
                                     200704,#768,         # 29 9
                                     200704,#4096,        # 15 5
                                     200704,#4096,        # 8 3
                                    ],
                         'idx': 0,
                         'dropout':0.2}
        
        
        model_card_7c3l = {'name': '7c3l', 'model': '7c3l', 'channels': 3, 'Ks': (3,5),
                          'f_lin_lay':[248832,    # 452 144 
                                    59904,        # 226 72
                                    11264,        # 113 36
                                    1536,         # 57 18 
                                    172032,       # 29 9
                                    172032,       # 15 5
                                    172032,       # 8 3
                                      ], 
                           'idx': 1,
                          'dropout':0.2}
        
        model_card_6c3l = {'name': '6c3l', 'model': '6c3l', 'channels': 3, 'Ks': (3,5),
                          'f_lin_lay':[272384,   # 452 144 
                                    71680,      # 226 72 
                                    16640,      # 113 36 
                                    3840,       # 57 18 
                                    1024,       # 29 9
                                    193024,     # 15 5 
                                    193024,     # 8 3
                                      ], 
                           'idx': 1,
                          'dropout':0.2}
        
        model_card_8c3l = {'name': '8c3l', 'model': '8c3l', 'channels': 3, 'Ks': (3,5),
                          'f_lin_lay':[248832,   # 452 144 
                                    59904,      # 226 72 
                                    14080,      # 113 36
                                    1536,       # 57 18 
                                    172032,     # 29 9 
                                    172032,     # 15 5  
                                    172032,     # 8 3
                                      ], 
                           'idx': 1,
                          'dropout':0.2}
        
        
        
        
        model_card_4c3l = {'name': '4c3l', 'model': '4c3l', 'channels': 3, 'Ks': (3,5),
                          'f_lin_lay':[539904,    # 452 144
                                     141056,     # 226 72 
                                     35840,     # 113 36  304640
                                     9984,      # 57 18
                                     2304,      # 29 9
                                     512,       # 15 5
                                     256],      # 8 3
                          'idx': 2,
                          'dropout':0.2}      
        
        model_card_3c2l = {'name': '3c2l', 'model': '3c2l', 'channels': 3, 'Ks': (3,5),
                          'f_lin_lay':[1069888,  # 452 144 
                                     274688,     # 226 72 
                                     68096,      # 113 36 
                                     17280,      # 57 18 
                                     3840,       # 29 9
                                     960,        # 15 5
                                     256],       # 8 3
                          'idx': 3,
                          'dropout':0.2}       
        
        model_card_2c2l = {'name': '2c2l', 'model': '2c2l', 'channels': 3, 'Ks': (3,5),
                          'f_lin_lay':[1055232 , # 452 144 
                                     267264,     # 226 72 
                                     64512,#     # 113 36   
                                     15552,      # 57 18 
                                     3072,       # 29 9
                                     640,        # 15 5
                                     128],       # 8 3
                          'idx': 4,
                          'dropout':0.2}       
        
        self.modelcards =[model_card_vgg,model_card_7c3l,model_card_6c3l,model_card_8c3l,model_card_4c3l,model_card_3c2l,model_card_2c2l]
        
        resolution_card_452144 = {'resolution':[452,144], 'padding':5, 'index':0}
        resolution_card_22672 = {'resolution':[226,72], 'padding':5, 'index':1}
        resolution_card_11336 = {'resolution':[113,36], 'padding':2, 'index':2}
        resolution_card_5715 = {'resolution':[57,18], 'padding':1, 'index':3}
        resolution_card_299 = {'resolution':[29,9], 'padding':0, 'index':4} # 
        resolution_card_155 = {'resolution':[15,5], 'padding':0, 'index':5}
        resolution_card_83 = {'resolution':[8,3], 'padding':0, 'index':6}
        
        self.resolutioncards =[resolution_card_452144, resolution_card_22672, resolution_card_11336, resolution_card_5715, resolution_card_299, resolution_card_155, resolution_card_83]

    def modname2linlay(self, model_name:str, res:list):
        for card in self.modelcards:
            if card['name']==model_name:
                print(card['name'], res)
                linlay = get_lin_lay(card, res)
                return linlay

    def res2pad(self, res):
        for r in self.resolution_cards:
            if r['resolution'] == res:
                return r['padding']


def get_lin_lay(model_card, resolution):
    if resolution == [452, 144]:
        lin_lay = model_card['f_lin_lay'][0]
    elif resolution == [226, 72]:
        lin_lay = model_card['f_lin_lay'][1]
    elif resolution == [113, 36]:
        lin_lay = model_card['f_lin_lay'][2]
    elif resolution == [57, 18]:
        lin_lay = model_card['f_lin_lay'][3]
    elif resolution == [29, 9]:
        lin_lay = model_card['f_lin_lay'][4]
    elif resolution == [15, 5]:
        lin_lay = model_card['f_lin_lay'][5]
    elif resolution == [8, 3]:
        lin_lay = model_card['f_lin_lay'][6]
    else:
        print("PARAMETER NOT FOUND: \n f_lin_lay FROM MODEL CARD")
    return lin_lay
    
    
        