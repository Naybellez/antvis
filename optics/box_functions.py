# Functions related to BOX API
# Allows up easy upload or download of folders and files to and from BOX online data storage

from boxsdk import OAuth2, Client
import os

class BoxData():

    def __init__(self, access_token):
        oauth =OAuth2(
            client_id= '', # Insert your BOX client ID
            client_secret ='', # Insert your BOX secret
            access_token = access_token)
    
        self.client = Client(oauth)
        user = self.client.user().get()
        print('Current User: ', user.id)
    
    def get_box_items(self, folderID):
        folder = self.client.folder(folder_id=folderID).get() # Return box folder
        print(f'Folder {folder.name} has {folder.item_collection["total_count"]} items in it') # Print the number i=of items in the folder
        self.items = self.client.folder(folder_id=folderID).get_items() # Return the items
        return self.items
    
    def download_files(self,datafolder, save_location):
        # Download files from BOX folder
        for idx, item in enumerate(datafolder):
            save_location = save_location # local location to save the files to
            item_content = self.client.file(item.id).get() # get the items
            if item.name not in save_location: # Prevents downloading items you already have - useful if download stops part way through
                with open(os.path.join(save_location, item.name), 'wb') as open_file:
                    item_content.download_to(open_file)
                    open_file.close()
	
    def upload_folder(self,folderID, og_fol:str):
        # Upload a local folder to BOX
        to_folder = self.client.folder(folder_id=folderID) # folder location ID on BOX
        self.items = self.client.folder(folder_id=folderID).get_items() # get items already in folder

        box_list = []
        [box_list.append(item.name) for item in self.items] # create list of items in BOX folder
        print(box_list)
        for f in os.listdir(og_fol):
            print('f',f)
            file_path = os.path.join(og_fol, f) # path to local folder
        
            try:
                if f not in box_list: # Prevent uploading an item already in box folder
                    print(f"uploading...:")
                    to_folder.upload(file_path, file_name=f)
                    
            except BoxAPIException:
                print(f" {f} already exists")
                continue

        folder = to_folder.get()
        print(f'Folder {folder.name} has {folder.item_collection["total_count"]} items in it') # print count of items in BOX folder
        print("Upload Complete")

    def upload_folder2(self, folderID, og_fol:str):
        # another upload function - fewer lines
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
        # allows upload to multiple folder in box based on file types
        # grph_folder : Graphs png/ jpg
        # pkl_folder : pickle files  - contain model_state_dict and simulation paramaters
        # csv_folder : csv file - contrain simulation parameters
        # json_folder : json file - contain simulation parameters
        from tqdm import tqdm
        box_list = []
        grph_folder = self.client.folder(folder_id=grphfolderID)
        pkl_folder = self.client.folder(folder_id=pklfolderID)
        csv_folder = self.client.folder(folder_id=csvfolderID)
        json_folder = self.client.folder(folder_id=jsonfolderID)
        
        folders = [grph_folder, pkl_folder, csv_folder, json_folder]

        # box_list to contain files already on BOX
        print('collating box files... ')
        for folder in folders:
            self.items = folder.get_items()
            box_list += [item.name for item in self.items]
            
        print('Uploading... ')
        for f in tqdm(os.listdir(og_fol)): # should put a tqm around this
            file_path = os.path.join(og_fol, f)
            if os.path.isfile(file_path):
                # handles sending file types to corresponding BOX folder
                try:
                    if f not in box_list:
                        file_type = f[-3:]
                        if file_type == 'pkl':
                            pkl_folder.upload(file_path, file_name=f)
                        elif file_type == 'csv':
                            csv_folder.upload(file_path, file_name=f)
                        elif file_type == 'son':
                            json_folder.upload(file_path, file_name=f)
                        elif file_type.lower() == 'png':
                            grph_folder.upload(file_path, file_name=f)
                    else:
                        pass
                except Exception as e:
                    print(f"Error uploading {f}: {str(e)}")
                except Exception as e:
                    print(f"Unexpected error uploading {f}: {str(e)}")
        print("Task Complete!")
                    
    def upload_files(self, folderID, file_path, file):
        # Uplaod on file by file bases not by folder
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
-F attributes='{"name":"AugmentedDS_IDSW/${FILE##*/}", "parent":{"id":""}}'\
-F file=@$FILE
done
"""

