import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
import numpy as np
from numpy import nan
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import csv
from sklearn.impute import KNNImputer
from zipfile import ZipFile
from os.path import basename

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '\\downloads\\'
ZIP_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '\\downloads\\zip\\'
REPORT_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '\\downloads\\report\\'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['ZIP_FOLDER'] = ZIP_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			global folder
			folder=UPLOAD_FOLDER+ filename
			red=pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			d=red.columns
			xfile=os.path.splitext(filename)[0]
			xxfile=xfile+".zip"
			#datapreprocessing(folder, filename)
			return render_template("home.html",result=d, fo=folder, file=filename, xxfile=xxfile)
			
			# return redirect(url_for('uploaded_file',
			#                         filename=filename))
	return render_template('home.html')

@app.route('/get', methods=['GET', 'POST'])
def get():
	radioValue=request.form['radioValue']
	folder=request.form['folder']
	global file
	file=request.form['file']
	datapreprocessing(folder, file, radioValue)
	xfile=os.path.splitext(file)[0]
	file=xfile+".zip"
	return send_from_directory(app.config['ZIP_FOLDER'],
							   file)
	

def datapreprocessing(path, filename, dataname):
	dataset = pd.read_csv(path, header=0)
	dataset=pd.DataFrame(dataset)
	print(dataset.eq(0).any().any())
	
	inDep_null=dataset.loc[:, dataset.columns != dataname]
	Dep_null=dataset.loc[:, dataset.columns == dataname]
	print(inDep_null.eq(0).any().any())

	if inDep_null.eq(0).any().any()==True:
		inDep_null = inDep_null.replace(0, nan)

	dataset = pd.concat([inDep_null, Dep_null], axis=1, sort=False)

	df_numerics_only = dataset.select_dtypes(include=np.number)
	def outlier_detect(df):
		for i in df.describe().columns:
			Q1=df.describe().at['25%',i]
			Q3=df.describe().at['75%',i]
			IQR=Q3 - Q1
			LTV=Q1 - 1.5 * IQR
			UTV=Q3 + 1.5 * IQR
			x=np.array(df[i])
			p=[]
			for j in x:
				if j < LTV or j>UTV:
					p.append(df[i].median())
				else:
					p.append(j)
			df[i]=p
		return df

	outlier_detect(df_numerics_only)
	colnames_outliers_numerics_only = df_numerics_only.select_dtypes(include=np.number).columns.tolist()
	dataset.loc[:,colnames_outliers_numerics_only] = df_numerics_only.loc[:,colnames_outliers_numerics_only]
	xfile=os.path.splitext(filename)[0]
	pfile=xfile+".pdf"
	table4 = plt.figure()
	ax=table4.add_subplot()

	cell_text = []
	for row in range(len(dataset)):
	    cell_text.append(dataset.iloc[row])

	ax.table(cellText=cell_text, colLabels=dataset.columns, loc='center')
	ax.axis('off')

	pdf = matplotlib.backends.backend_pdf.PdfPages(REPORT_FOLDER + pfile)
	pdf.savefig(table4,bbox_inches='tight')

	if dataset.duplicated().sum()>0:
		dataset=dataset.drop_duplicates(keep="first")
	table1 = plt.figure()
	ax=table1.add_subplot()

	cell_text = []
	for row in range(len(dataset)):
	  cell_text.append(dataset.iloc[row])

	ax.table(cellText=cell_text, colLabels=dataset.columns, loc='center')
	ax.axis('off')
	pdf.savefig(table1,bbox_inches='tight')
	B=dataset.loc[:,:].values
	df_numerics_only = dataset.select_dtypes(include=np.number)
	df_category_only = dataset.select_dtypes(include=np.object)
	
	if df_numerics_only.isnull().sum().sum()>0:
		if len(dataset)<10000:
			for i in df_numerics_only:
				colnames_numerics_only = dataset.select_dtypes(include=np.number).columns.tolist()
			ind=[]
			for col in colnames_numerics_only:
				if col != dataname:
					ind.append(dataset.columns.get_loc(col))
			imputer = KNNImputer(n_neighbors=2, weights="uniform")
			B[:,ind]=imputer.fit_transform(B[:,ind])
	
  
		else:
			for i in df_numerics_only:
				colnames_numerics_only = dataset.select_dtypes(include=np.number).columns.tolist()
			indexes=[]
			for col in colnames_numerics_only:
				if col != dependent:
					indexes.append(dataset.columns.get_loc(col))
			imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
			imputer.fit(B[:,indexes])
			B[:,indexes]=imputer.transform(B[:,indexes])
	
	df1=pd.DataFrame(B)
	table2 = plt.figure()
	ax=table2.add_subplot()

	cell_text = []
	for row in range(len(df1)):
	    cell_text.append(df1.iloc[row])

	ax.table(cellText=cell_text, colLabels=df1.columns, loc='center')
	ax.axis('off')
	pdf.savefig(table2,bbox_inches='tight')


	df_category_only = dataset.select_dtypes(include=np.object)
	if df_category_only.isnull().sum().sum()>0:
		for i in df_category_only:
			colnames_category_only = dataset.select_dtypes(include=np.object).columns.tolist()
		Cindexes=[]
		for col in colnames_category_only:
			if col != dataname:
				Cindexes.append(dataset.columns.get_loc(col))
		imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
		imputer.fit(B[:,Cindexes])
		B[:,Cindexes]=imputer.transform(B[:,Cindexes])
	df2=pd.DataFrame(B)
	table3 = plt.figure()
	ax=table3.add_subplot()

	cell_text = []
	for row in range(len(df2)):
	    cell_text.append(df2.iloc[row])

	ax.table(cellText=cell_text, colLabels=df2.columns, loc='center')
	ax.axis('off')
	pdf.savefig(table3,bbox_inches='tight')

	Cindexes=[]
	for col in dataset:
		if col == dataname:
			Cindexes.append(dataset.columns.get_loc(col))
	Y=df2.iloc[:,Cindexes].values

	dep=dataset[[dataname]]
	df_category_for_dependent = dep.select_dtypes(include=np.object)
	lable_value=df_category_for_dependent.shape[1]

	indexess=[]
	for col in dataset:
		if col != dataname:
			indexess.append(dataset.columns.get_loc(col))
	X=df2.iloc[:,indexess].values
	
	for i in df_category_only:
		colnames_category_only = dataset.select_dtypes(include=np.object).columns.tolist()
	Cindexes=[]
	for col in colnames_category_only:
		if col != dataname:
			Cindexes.append(dataset.columns.get_loc(col))
	print(Cindexes)
	ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), Cindexes)], remainder='passthrough')
	X = np.array(ct.fit_transform(X))
	
	df_x=pd.DataFrame(X)

	if lable_value==1:
		le = LabelEncoder()
		Y= le.fit_transform(Y)

	df_y=pd.DataFrame(Y)


	result = pd.concat([df_x, df_y], axis=1, sort=False)
	
	B=result.iloc[:,:].values

	df3=pd.DataFrame(B)
	table4 = plt.figure()
	ax=table4.add_subplot()

	cell_text = []
	for row in range(len(df3)):
	    cell_text.append(df3.iloc[row])

	ax.table(cellText=cell_text, colLabels=df3.columns, loc='center')
	ax.axis('off')
	pdf.savefig(table4,bbox_inches='tight')
	pdf.close()

	
	myFile = open(app.config['DOWNLOAD_FOLDER'] + filename, 'w', newline='')
	writer = csv.writer(myFile)
	with myFile:
		writer.writerows(B)
	with ZipFile(ZIP_FOLDER+xfile+".zip", 'w') as zipObj2:
	   # Add multiple files to the zip
	   xxfile=os.path.splitext(filename)[0]
	   pdffilename=xxfile+".pdf"
	   zipObj2.write("./downloads/"+filename)
	   zipObj2.write("./downloads/report/"+pdffilename)
	   zipObj2.close()

@app.route('/visualize')
def visualize():
	global folder
	global file
	dataset = pd.read_csv(folder)
	ax = dataset.plot.box()  # s is an instance of Series
	fig = ax.get_figure()
	boximg= 'static/images/'+file+'box.png'
	fig.savefig(boximg ,dpi=300,bbox_inches="tight")

	ax1 = dataset.plot.barh(stacked=True)  # s is an instance of Series
	fig1 = ax1.get_figure()
	stackimg= 'static/images/'+file+'stacked.png'
	fig1.savefig(stackimg,dpi=300,bbox_inches="tight")

	ax2 = dataset.plot.area()  # s is an instance of Series
	fig2 = ax2.get_figure()
	areaimg= 'static/images/'+file+'area.png'
	fig2.savefig(areaimg,dpi=300,bbox_inches="tight")

	ax3 = dataset.plot.kde()  # s is an instance of Series
	fig3 = ax3.get_figure()
	lineimg= 'static/images/'+file+'line.png'
	fig3.savefig(lineimg,dpi=300,bbox_inches="tight")
	return render_template("visualize.html",box=boximg, stack=stackimg, area=areaimg, line=lineimg )

@app.route('/downloads/<filename>')
def uploaded_file(filename):
	xfile=os.path.splitext(filename)[0]
	filename=xfile+".zip"
	return send_from_directory(app.config['ZIP_FOLDER'], filename)