
""" STYLESHEET """
styles = """
QListWidget, QListView, QTreeWidget, QTreeView {
    outline: 0px;
}
QListWidget {
    min-width: 120px;
    max-width: 120px;
    color: white;
    background: #007267;
    font-weight: 500;
    font-size: 18px;
}
QListWidget::item:selected {
    background: rgb(16,100,112);
    border-left: 3px solid #00ACB8;
    color: white;
}
HistoryPanel::item:hover {background: rgb(52, 52, 52);}


/* QStackedWidget {background: rgb(30, 30, 30);} */


QCheckBox {
    margin-right: 50px;
    spacing: 5px;
    font-size: 18px;    
}

QCheckBox::indicator {
    width:  27px;
    height: 27px;
}

QProgressBar {
text-align: center;
border: solid grey;
border-radius: 7px;
color: white;
background: #ddd;
font-size: 20px;
}
QProgressBar::chunk {
background-color: #00ACB8;
border-radius :7px;

}      

QPushButton {
font-size: 16px; 
font-weight: 600; 
padding: 8px; 
background: #007267; 
color: white; 
border-radius: 7px;
}

QLineEdit {
font-size: 16px; 
padding: 8px; 
font-weight: 400; 
background: #ddd; 
border-radius: 7px; 
margin-bottom: 5px;
}

QComboBox {
    font-size: 16px; 
    padding: 8px; 
    font-weight: 400; 
    background: #ddd; 
    border-radius: 7px; 
    margin-bottom: 5px;
}

QComboBox::drop-down {
    border: 2px; 
}


QComboBox QAbstractItemView {
    font-size: 16px; 
    border: 0 !important; 
    outline: none !important; 
    color: #007267;
    font-weight: 400; 
    border-radius: 7px; 
}

QLabel {
font-size: 20px; 
font-weight: bold; 
padding-top: 5px; 
padding-bottom: 10px;
color: black;
}

QRadioButton {
font-size: 17px; 
font-weight: 500; 
padding-top: 6px;
margin:  0px;
}

QToolButton {
border: 0px;
outline: none:
background: #f0f0f0;   
}

"""
# QToolBar {
# 	background-color: rgb(230, 230, 230);
# 	spacing: 3px;
# 	padding: 2px;
# }
#
# QMenu {
# background-color: rgb(230, 230, 230);
# spacing: 3px;
# padding: 2px;
# color: black;
# }