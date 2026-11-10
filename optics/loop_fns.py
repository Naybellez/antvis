
import random
import torch
from tqdm import tqdm
import sys
sys.path.append('../.')
from functions import ImageProcessor,label_oh_tf
from modelManagment import choose_scheduler

# loops                     

def loop(model, X, Y, loss_fn, device, size, pad, num_classes, model_name, av_lum,colour= 'colour', optimizer =None, scheduler= None, train =True):	# Train and Val loops. Default is train
    model = model
    total_samples = len(X)
    if train:
        model.train()
    else:
        model.eval()

    predict_list = []
    total_count = 0
    num_correct = 0
    current_loss = 0

    for idx, img in enumerate(X):
        prepro = ImageProcessor(device)
        if model_name == 'vgg16':
            tense = prepro.colour_size_tense(img, colour, size,av_lum, pad, vg=True) #[29, 9], 15, 5, [8,3]
        elif (model_name == '7c3l' and size == [29, 9]) or (model_name == '7c3l' and size == [15, 5]) or (model_name == '7c3l' and size ==[8, 3]):
            tense = prepro.colour_size_tense(img, colour, size, av_lum, pad, vg=True) #img_path, col:str, size, av_lum,  pad:int
        else:
            tense = prepro.colour_size_tense(img, colour, size,av_lum, pad)

        prediction = model.forward(tense)
        label = label_oh_tf(Y[idx], num_classes).to(device)

        loss = loss_fn(prediction, label)
        predict_list.append(prediction.argmax())
        if prediction.argmax() == label.argmax():
            num_correct +=1
   
        total_count+=1
        current_loss += loss.item()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if scheduler:
        scheduler.step()
    if train:
        return current_loss, predict_list, num_correct, model, optimizer #, lr_ls
    else:
        return current_loss, predict_list, num_correct



def test_loop(model, X, Y, loss_fn, device, title, col_dict, num_classes=11):
	model = model.eval()
	predict_list = []
	total_count =0
	num_correct = 0
	correct = 0
	colour = col_dict['colour']
	size = col_dict['size']

	with torch.no_grad():
		for idx, img in enumerate(X):
			prepro = ImageProcessor(device)
			tense = prepro.colour_size_tense(img, colour, size)
			prediction = model.forward(tense)
			label = label_oh_tf(Y[idx], device, num_classes)

			if prediction.argmax()==label.argmax():
				num_correct +=1
			total_count +=1
			correct +=(prediction.argmax()==label.argmax()).sum().item()

		acc = num_correct/total_count
		accuracy = 100*(acc)

		X = list(X)
		log_test_score(acc, accuracy, X)


def loop_og(model, X, Y, loss_fn, device, col_dict, num_classes, model_name, optimizer =None, scheduler= None, train =True):	# Train and Val loops. Default is train
    model = model
    total_samples = len(X)
    if train:
        model.train()
        #lr_ls = []
    else:
        model.eval()

    predict_list = []
    total_count = 0
    num_correct = 0
    current_loss = 0
    colour = col_dict['colour']
    size = col_dict['size']
    pad = col_dict['pad']
    av_lum = col_dict['av_lum']

    for idx, img in enumerate(X):
        prepro = ImageProcessor(device)
        tense = prepro.colour_size_tense(img, colour, size, av_lum, pad)

        prediction = model.forward(tense)
        label = label_oh_tf(Y[idx], num_classes).to(device)
       
        loss = loss_fn(prediction, label)
        predict_list.append(prediction.argmax())

        if prediction.argmax() == label.argmax():
            num_correct +=1
           
        total_count+=1
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
        current_loss += loss.item()
    if train:
        return current_loss, predict_list, num_correct, model, optimizer #, lr_ls
    else:
        return current_loss, predict_list, num_correct



def test_loop_og(model, X, Y, loss_fn, device, col_dict,title, num_classes):
    model = model.eval()
    predict_list = []
    total_count =0
    num_correct = 0
    correct = 0
    colour = col_dict['colour']
    size = col_dict['size']

    with torch.no_grad():
        for idx, img in enumerate(X):
            prepro = ImageProcessor(device)
            tense = prepro.colour_size_tense(img, colour, size)
            prediction = model.forward(tense)
            label = label_oh_tf(Y[idx], device, num_classes)

            if prediction.argmax()==label.argmax():
                num_correct +=1
            total_count +=1
            correct +=(prediction.argmax()==label.argmax()).sum().item()

        acc = num_correct/total_count
        accuracy = 100*(acc)

        X = list(X)
        log_test_score(acc, accuracy, X)




def loop_batch(model, data, loss_fn, batch_size, sample,random_value,epoch,loop_run_name, save_dict, device, optimizer =None, scheduler= None, train =True):	# Train and Val loops. Default is train
    
    model = model #.
    total_samples = len(data)
    if optimizer: # need a choose scheduler function!
        print("Optimizer present: ",optimizer)
        scheduler = choose_scheduler(save_dict, optimizer)
    if train:
        model.train()
    else:
        model.eval()   
    predict_list = []
    total_count = 0
    num_correct = 0
    current_loss = 0
    labels =[]
    #print("loopBatch pre loop- Current allocated memory (GB):", torch.cuda.memory_allocated(device=device) / 1024 ** 3)

    for i, batch in enumerate(data,0):

        x_batch, y_batch = batch
        
        prediction = model.forward(x_batch.to(device))
        loss = loss_fn(prediction, y_batch.to(device))
        #print("loss calculated- Current allocated memory (GB):", torch.cuda.memory_allocated(device=device) / 1024 ** 3)
        

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        for i in range(len(y_batch)-1):
            if y_batch[i].argmax() == prediction[i].argmax():
                num_correct +=1
                

        [predict_list.append(pred.argmax().item()) for pred in prediction]
        [labels.append(y.argmax().item()) for y in y_batch]

        total_count+= batch_size
        current_loss += loss.item()

        #print("loopBatch end of loop Current allocated memory (GB):", torch.cuda.memory_allocated(device=device) / 1024 ** 3)
        if scheduler:
            scheduler.step(loss)

    if train:
        return current_loss, predict_list, labels, num_correct, model, optimizer #, lr_ls
    else:
        return current_loss, predict_list, labels, num_correct # changed y_batch to labels in return 



def test_loop_batch(model,data, loss_fn, batch_size, device):
    import sys
    sys.path.append('../.')
    model = model.eval()
    predict_list = []
    label_list = []
    total_count =0
    num_correct = 0
    correct = 0

    with torch.no_grad():
        for i, batch in enumerate(data,0):
            tense, label = batch
            print("in test batch. got tense and label from batch. type len.  tese:", type(tense), len(tense), " label:", type(label), len(label))
            print(label)

            
            prediction = model.forward(tense.to(device))
            for i in range(len(label)-1):
                if label[i].argmax() == prediction[i].argmax():
                    num_correct +=1
            [predict_list.append(pred.argmax().to('cpu').item()) for pred in prediction]
            [label_list.append(lab.argmax().to('cpu').item()) for lab in label]
            print("in test bAtch post list comprehension. pred:", len(predict_list), "lab:", len(label_list))
            total_count += batch_size
        acc = num_correct/total_count
        accuracy = 100*(acc)
        print(accuracy)
        return accuracy, predict_list, label_list

def train_val_batch(model, train, val, loop_run_name, save_dict, lr, loss_fn, epochs, batch_size, optimizer, scheduler_value, device): #train_dl, val_dl, 
    import sys
    sys.path.append('../.')
    import pickle
    #import wandb
    from IPython.display import clear_output
    model.train()
    t_loss_list = []
    v_loss_list = []
    t_predict_list = []
    v_predict_list = []
    t_accuracy_list = []
    v_accuracy_list = []
    t_label_list = []
    v_label_list = []
    #labels = []
    sample = False
    
    total_epochs = 0
    print("Before Epochs of training - Current allocated memory (GB):", torch.cuda.memory_allocated(device=device) / 1024 ** 3)
    for epoch in tqdm(range(save_dict['start_epoch'],epochs)):

        random_value = random.randrange(0,batch_size)
        print('Training...')

        t_loss, train_prediction, t_label_list, t_correct, model, optimizer = loop_batch(model, train, loss_fn, batch_size,sample,random_value,epoch,loop_run_name, save_dict, device, optimizer =optimizer, scheduler= scheduler_value, train =True) #, scheduler =scheduler
        print('training..  2')
        #!nvidia-smi
        
        t_loss_list.append(t_loss)
        t_predict_list.append(train_prediction)
    
        train_acc = (t_correct/(len(train)*batch_size)*100) ###
        print('train accuracy: ', train_acc )
        t_accuracy_list.append(train_acc)

        print('validating...')
        #!nvidia-smi
        
        v_loss, val_prediction, v_label_list, val_correct= loop_batch(model, val, loss_fn, batch_size,sample,random_value,epoch,loop_run_name, save_dict, device, optimizer =None, scheduler= None, train =False)
        v_loss_list.append(v_loss)
        v_predict_list.append(val_prediction)
        
        val_acc = (val_correct/(len(val)*batch_size)*100)
        v_accuracy_list.append(val_acc)
        print('validation accuracy: ', val_acc )
    
        total_epochs += 1
        print(f"After Epoch {total_epochs} - Current allocated memory (GB):", torch.cuda.memory_allocated() / 1024 ** 3)
        #if epoch %50==0 and epoch !=0 and epoch != int(save_dict['start_epoch']):
        #    from plotting import learning_curve, accuracy_curve
        #    #checkpoint = copy.deepcopy(model)
        #    checkpoint_id = f"{save_dict['model']}_{save_dict['optimiser']}_{save_dict['sched']}_{epoch}E_{save_dict['res']}_seed{save_dict['seed']}"
        #    torch.save(model.state_dict(), str(save_dict['checkpoint_save_loc'])+checkpoint_id+".pkl", pickle_module=pickle)
        #    learning_curve(t_loss_list, v_loss_list, save_location=str(save_dict['checkpoint_save_loc']),run_name=checkpoint_id)
        #    accuracy_curve(t_accuracy_list, v_accuracy_list,save_location=str(save_dict['checkpoint_save_loc']),run_name=checkpoint_id)
            #plot_confusion(predictions= test_predict_list, actual= y_test, title = "Test Confusion matrix", run_name = checkpoint_id,save_location =checkpoint_saveloc)
        clear_output()
        
    save_dict['Current_Epoch'] = epochs
    save_dict['training_samples'] = len(train)
    save_dict['validation_samples'] = len(val)
    
    save_dict['t_accuracy_list'] = t_accuracy_list 
    save_dict['v_accuracy_list'] = v_accuracy_list  #

    save_dict['t_loss_list'] = t_loss_list
    save_dict['v_loss_list'] = v_loss_list
    
    save_dict['t_labels'] = t_label_list
    save_dict['v_labels'] = v_label_list
    
    save_dict['t_predict_list'] = t_predict_list 
    save_dict['v_predict_list'] = v_predict_list  #
    
    return model, save_dict

from functions import ImageProcessor