import webbrowser

def open(keyword):
	url = 'https://www.google.co.jp/search?q='+keyword.replace(' ','+')+'&source=lnms&tbm=isch'
	webbrowser.open(url,autoraise=True)