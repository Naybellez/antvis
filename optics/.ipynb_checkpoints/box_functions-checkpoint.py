from boxsdk import OAuth2, Client
import os

class BoxData():

    def __init__(self, access_token):
        oauth =OAuth2(
            client_id= 'hw534w4beg3mscd8v265vpkk8ndzc5y1',
            client_secret ='nmL4fcHjb2drntVJxGSqQvjt19t0hIlu',
            access_token = access_token)
    
    
        self.client = Client(oauth)
        user = self.client.user().get()
        print('Current User: ', user.id)
    
    def get_box_items(self, folderID):
        folder = self.client.folder(folder_id=folderID).get()
        print(f'Folder {folder.name} has {folder.item_collection["total_count"]} items in it')
        self.items = self.client.folder(folder_id=folderID).get_items()
        return self.items
    
    def download_files(self,datafolder, save_location):
        for idx, item in enumerate(datafolder):
            save_location = save_location
            item_content = self.client.file(item.id).get()
            if item.name not in save_location:
                with open(os.path.join(save_location, item.name), 'wb') as open_file:
                    item_content.download_to(open_file)
                    open_file.close()
	
    def upload_folder(self,folderID, og_fol:str):
        to_folder = self.client.folder(folder_id=folderID)#.get #?
        self.items = self.client.folder(folder_id=folderID).get_items()
        #print(self.items)
        #item_content = self.client.file(item.id).get() 
        #print(item_content)
        box_list = []
        #[box_list.append(self.client.file(item.id).get()) for item in self.items] 
        [box_list.append(item.name) for item in self.items]
        print(box_list)
        for f in os.listdir(og_fol):
            print('f',f)
            file_path = os.path.join(og_fol, f)
        
            #print(item_content)
            #try:
            #print(self.items)
            try:
                if f not in box_list: #box_list:
                    print(f"uploading...:")
                    to_folder.upload(file_path, file_name=f)
                    
            except BoxAPIException:
                print(f" {f} already exists")
                continue

        folder = to_folder.get()
        print(f'Folder {folder.name} has {folder.item_collection["total_count"]} items in it')
        print("Upload Complete")

    def upload_folder2(self, folderID, og_fol:str):
        to_folder = self.client.folder(folder_id=folderID)
        self.items = self.client.folder(folder_id=folderID).get_items()
        box_list = [item.name for item in self.items]
        
        for f in os.listdir(og_fol):
            file_path = os.path.join(og_fol, f)
            if os.path.isfile(file_path):
                try:
                    if f not in box_list:
                        print(f"Uploading {f}...")
                        to_folder.upload(file_path, file_name=f)
                    else:
                        print(f"{f} already exists in Box. Skipping.")
                except Exception as e:
                    print(f"Error uploading {f}: {str(e)}")
                except Exception as e:
                    print(f"Unexpected error uploading {f}: {str(e)}")

    def upload_folder3(self, pklfolderID, csvfolderID, grphfolderID, jsonfolderID, og_fol:str):
        # destination folders in box
        from tqdm import tqdm
        box_list = []
        grph_folder = self.client.folder(folder_id=grphfolderID)
        pkl_folder = self.client.folder(folder_id=pklfolderID)
        csv_folder = self.client.folder(folder_id=csvfolderID)
        json_folder = self.client.folder(folder_id=jsonfolderID)
        
        folders = [grph_folder, pkl_folder, csv_folder, json_folder]
        
        print('collating box files... ')
        for folder in folders:
            self.items = folder.get_items()
            box_list += [item.name for item in self.items]
            
        print('Uploading... ')
        for f in tqdm(os.listdir(og_fol)): # should put a tqm around this
            file_path = os.path.join(og_fol, f)
            if os.path.isfile(file_path):
                try:
                    if f not in box_list:
                        file_type = f[-3:]
                        if file_type == 'pkl':
                            #print(f"Uploading pkl    {f}  ...")
                            pkl_folder.upload(file_path, file_name=f)
                        elif file_type == 'csv':
                            #print(f"Uploading csv    {f}  ...")
                            csv_folder.upload(file_path, file_name=f)
                        elif file_type == 'son':
                            #print(f"Uploading json    {f}  ...")
                            json_folder.upload(file_path, file_name=f)
                        elif file_type.lower() == 'png':
                            #print(f"Uploading image    {f}  ...")
                            grph_folder.upload(file_path, file_name=f)
                    else:
                        print(f"{f} already exists in Box.     Skipping.")
                except Exception as e:
                    print(f"Error uploading {f}: {str(e)}")
                except Exception as e:
                    print(f"Unexpected error uploading {f}: {str(e)}")
        print("Task Complete!")
                    
    def upload_files(self, folderID, file_path, file):
        to_folder = self.client.folder(folder_id=folderID)
        self.items = self.client.folder(folder_id=folderID).get_items()
        box_list = [item.name for item in self.items]
        
        try:
            if file not in box_list: #box_list:
                print(f"uploading...:")
                to_folder.upload(file_path+'/'+file)
                
        except Exception as e:
            print(f" {file} already exists: {str(e)}")
            
            
        folder = to_folder.get()
        print(f'Folder {folder.name} has {folder.item_collection["total_count"]} items in it')
        print("Upload Complete")

        
			
	


# box upload from commandline
"""
for FILE in /optics/AugmentedDS_IDSW/*; do
curl https://upload.box.com/api/2.0/file/content \
-H "Authorization: Bearer 34863846"\
-F attributes='{"name":"AugmentedDS_IDSW/${FILE##*/}", "parent":{"id":"623523"}}'\
-F file=@$FILE
done
"""

