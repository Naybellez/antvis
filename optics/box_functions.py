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

