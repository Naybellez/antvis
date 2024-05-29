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

			#except:
			#	print(f"uploading...:")
			#	to_folder.upload(file_path, file_name=f)

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

