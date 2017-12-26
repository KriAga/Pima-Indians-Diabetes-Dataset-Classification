{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from matplotlib.colors import ListedColormap\n",
    "import matplotlib.pyplot as plt\n",
    "def plot_decision_regions(X, y, classifier,test_idx=None, resolution=0.02):\n",
    "    # setup marker generator and color map\n",
    "    markers = ('s', 'x', 'v', '^', 'o')\n",
    "    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')\n",
    "    cmap = ListedColormap(colors[:len(np.unique(y))])\n",
    "    # plot the decision surface\n",
    "    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1\n",
    "    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1\n",
    "    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),\n",
    "                           np.arange(x2_min, x2_max, resolution))\n",
    "    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)\n",
    "    Z = Z.reshape(xx1.shape)\n",
    "    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)\n",
    "    plt.xlim(xx1.min(), xx1.max())\n",
    "    plt.ylim(xx2.min(), xx2.max())\n",
    "    # plot all samples\n",
    "    X_test, y_test = X[test_idx, :], y[test_idx]\n",
    "    for idx, cl in enumerate(np.unique(y)):\n",
    "        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],\n",
    "                    alpha=0.8, c=cmap(idx),\n",
    "                    marker=markers[idx], label=cl)\n",
    "    # highlight test samples\n",
    "    if test_idx:\n",
    "        X_test, y_test = X[test_idx, :], y[test_idx]\n",
    "        plt.scatter(X_test[:, 0], X_test[:, 1], c='',\n",
    "                    alpha=1.0, linewidth=1, marker='v',\n",
    "                    s=55, label='test set')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pregnancies</th>\n",
       "      <th>Glucose</th>\n",
       "      <th>BloodPressure</th>\n",
       "      <th>SkinThickness</th>\n",
       "      <th>Insulin</th>\n",
       "      <th>BMI</th>\n",
       "      <th>DiabetesPedigreeFunction</th>\n",
       "      <th>Age</th>\n",
       "      <th>Outcome</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>85</td>\n",
       "      <td>66</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "      <td>26.6</td>\n",
       "      <td>0.351</td>\n",
       "      <td>31</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>89</td>\n",
       "      <td>66</td>\n",
       "      <td>23</td>\n",
       "      <td>94</td>\n",
       "      <td>28.1</td>\n",
       "      <td>0.167</td>\n",
       "      <td>21</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>5</td>\n",
       "      <td>116</td>\n",
       "      <td>74</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>25.6</td>\n",
       "      <td>0.201</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>10</td>\n",
       "      <td>115</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>35.3</td>\n",
       "      <td>0.134</td>\n",
       "      <td>29</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>110</td>\n",
       "      <td>92</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>37.6</td>\n",
       "      <td>0.191</td>\n",
       "      <td>30</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \\\n",
       "0            1       85             66             29        0  26.6   \n",
       "1            1       89             66             23       94  28.1   \n",
       "2            5      116             74              0        0  25.6   \n",
       "3           10      115              0              0        0  35.3   \n",
       "4            4      110             92              0        0  37.6   \n",
       "\n",
       "   DiabetesPedigreeFunction  Age  Outcome  \n",
       "0                     0.351   31        0  \n",
       "1                     0.167   21        0  \n",
       "2                     0.201   30        0  \n",
       "3                     0.134   29        0  \n",
       "4                     0.191   30        0  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas\n",
    "diabetes = pandas.read_csv(\"D:/Vit/Semester_5/MachineLearning/Lab/pima-indians-diabetes-database/diabetes.csv\")\n",
    "diabetes.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',\n",
       "       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "diabetes.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\linear_model\\stochastic_gradient.py:84: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.\n",
      "  \"and default tol will be 1e-3.\" % type(self), FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import SGDClassifier\n",
    "ppn = SGDClassifier(loss='perceptron')\n",
    "lr = SGDClassifier(loss='log')\n",
    "svm = SGDClassifier(loss='hinge')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 768 entries, 0 to 767\n",
      "Data columns (total 9 columns):\n",
      "Pregnancies                 768 non-null int64\n",
      "Glucose                     768 non-null int64\n",
      "BloodPressure               768 non-null int64\n",
      "SkinThickness               768 non-null int64\n",
      "Insulin                     768 non-null int64\n",
      "BMI                         768 non-null float64\n",
      "DiabetesPedigreeFunction    768 non-null float64\n",
      "Age                         768 non-null int64\n",
      "Outcome                     768 non-null int64\n",
      "dtypes: float64(2), int64(7)\n",
      "memory usage: 54.1 KB\n"
     ]
    }
   ],
   "source": [
    "diabetes.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "X = diabetes[['Glucose','BMI']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Glucose</th>\n",
       "      <th>BMI</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>85</td>\n",
       "      <td>26.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>89</td>\n",
       "      <td>28.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>116</td>\n",
       "      <td>25.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>115</td>\n",
       "      <td>35.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>110</td>\n",
       "      <td>37.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>139</td>\n",
       "      <td>27.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>103</td>\n",
       "      <td>43.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>126</td>\n",
       "      <td>39.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>99</td>\n",
       "      <td>35.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>97</td>\n",
       "      <td>23.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>145</td>\n",
       "      <td>22.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>117</td>\n",
       "      <td>34.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>109</td>\n",
       "      <td>36.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>88</td>\n",
       "      <td>24.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>92</td>\n",
       "      <td>19.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>122</td>\n",
       "      <td>27.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>103</td>\n",
       "      <td>24.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>138</td>\n",
       "      <td>33.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>180</td>\n",
       "      <td>34.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>133</td>\n",
       "      <td>40.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>106</td>\n",
       "      <td>22.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>159</td>\n",
       "      <td>27.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>146</td>\n",
       "      <td>29.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>71</td>\n",
       "      <td>28.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>105</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>103</td>\n",
       "      <td>19.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>101</td>\n",
       "      <td>24.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>88</td>\n",
       "      <td>24.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>150</td>\n",
       "      <td>34.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>73</td>\n",
       "      <td>23.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>738</th>\n",
       "      <td>149</td>\n",
       "      <td>29.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>739</th>\n",
       "      <td>130</td>\n",
       "      <td>28.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>740</th>\n",
       "      <td>120</td>\n",
       "      <td>28.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>741</th>\n",
       "      <td>174</td>\n",
       "      <td>44.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>742</th>\n",
       "      <td>102</td>\n",
       "      <td>39.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>743</th>\n",
       "      <td>120</td>\n",
       "      <td>42.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>744</th>\n",
       "      <td>140</td>\n",
       "      <td>32.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>745</th>\n",
       "      <td>147</td>\n",
       "      <td>49.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>746</th>\n",
       "      <td>187</td>\n",
       "      <td>36.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>747</th>\n",
       "      <td>162</td>\n",
       "      <td>24.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>748</th>\n",
       "      <td>136</td>\n",
       "      <td>31.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>749</th>\n",
       "      <td>181</td>\n",
       "      <td>43.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>750</th>\n",
       "      <td>154</td>\n",
       "      <td>32.4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>751</th>\n",
       "      <td>128</td>\n",
       "      <td>36.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>752</th>\n",
       "      <td>123</td>\n",
       "      <td>36.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>753</th>\n",
       "      <td>190</td>\n",
       "      <td>35.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>754</th>\n",
       "      <td>170</td>\n",
       "      <td>44.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>755</th>\n",
       "      <td>126</td>\n",
       "      <td>30.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>756</th>\n",
       "      <td>78</td>\n",
       "      <td>31.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>757</th>\n",
       "      <td>189</td>\n",
       "      <td>30.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>758</th>\n",
       "      <td>166</td>\n",
       "      <td>25.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>759</th>\n",
       "      <td>100</td>\n",
       "      <td>30.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>760</th>\n",
       "      <td>118</td>\n",
       "      <td>45.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>761</th>\n",
       "      <td>107</td>\n",
       "      <td>29.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>762</th>\n",
       "      <td>187</td>\n",
       "      <td>37.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>763</th>\n",
       "      <td>114</td>\n",
       "      <td>32.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>764</th>\n",
       "      <td>109</td>\n",
       "      <td>32.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>765</th>\n",
       "      <td>100</td>\n",
       "      <td>32.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>766</th>\n",
       "      <td>122</td>\n",
       "      <td>49.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>767</th>\n",
       "      <td>163</td>\n",
       "      <td>39.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>768 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Glucose   BMI\n",
       "0         85  26.6\n",
       "1         89  28.1\n",
       "2        116  25.6\n",
       "3        115  35.3\n",
       "4        110  37.6\n",
       "5        139  27.1\n",
       "6        103  43.3\n",
       "7        126  39.3\n",
       "8         99  35.4\n",
       "9         97  23.2\n",
       "10       145  22.2\n",
       "11       117  34.1\n",
       "12       109  36.0\n",
       "13        88  24.8\n",
       "14        92  19.9\n",
       "15       122  27.6\n",
       "16       103  24.0\n",
       "17       138  33.2\n",
       "18       180  34.0\n",
       "19       133  40.2\n",
       "20       106  22.7\n",
       "21       159  27.4\n",
       "22       146  29.7\n",
       "23        71  28.0\n",
       "24       105   0.0\n",
       "25       103  19.4\n",
       "26       101  24.2\n",
       "27        88  24.4\n",
       "28       150  34.7\n",
       "29        73  23.0\n",
       "..       ...   ...\n",
       "738      149  29.3\n",
       "739      130  28.4\n",
       "740      120  28.4\n",
       "741      174  44.5\n",
       "742      102  39.5\n",
       "743      120  42.3\n",
       "744      140  32.7\n",
       "745      147  49.3\n",
       "746      187  36.4\n",
       "747      162  24.3\n",
       "748      136  31.2\n",
       "749      181  43.3\n",
       "750      154  32.4\n",
       "751      128  36.5\n",
       "752      123  36.3\n",
       "753      190  35.5\n",
       "754      170  44.0\n",
       "755      126  30.1\n",
       "756       78  31.0\n",
       "757      189  30.1\n",
       "758      166  25.8\n",
       "759      100  30.0\n",
       "760      118  45.8\n",
       "761      107  29.6\n",
       "762      187  37.7\n",
       "763      114  32.8\n",
       "764      109  32.5\n",
       "765      100  32.9\n",
       "766      122  49.7\n",
       "767      163  39.0\n",
       "\n",
       "[768 rows x 2 columns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "y = diabetes['Outcome']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0      0\n",
       "1      0\n",
       "2      0\n",
       "3      0\n",
       "4      0\n",
       "5      0\n",
       "6      0\n",
       "7      0\n",
       "8      0\n",
       "9      0\n",
       "10     0\n",
       "11     0\n",
       "12     0\n",
       "13     0\n",
       "14     0\n",
       "15     0\n",
       "16     0\n",
       "17     0\n",
       "18     0\n",
       "19     0\n",
       "20     0\n",
       "21     0\n",
       "22     0\n",
       "23     0\n",
       "24     0\n",
       "25     0\n",
       "26     0\n",
       "27     0\n",
       "28     0\n",
       "29     0\n",
       "      ..\n",
       "738    1\n",
       "739    1\n",
       "740    1\n",
       "741    1\n",
       "742    1\n",
       "743    1\n",
       "744    1\n",
       "745    1\n",
       "746    1\n",
       "747    1\n",
       "748    1\n",
       "749    1\n",
       "750    1\n",
       "751    1\n",
       "752    1\n",
       "753    1\n",
       "754    1\n",
       "755    1\n",
       "756    1\n",
       "757    1\n",
       "758    1\n",
       "759    1\n",
       "760    1\n",
       "761    1\n",
       "762    1\n",
       "763    1\n",
       "764    1\n",
       "765    1\n",
       "766    1\n",
       "767    1\n",
       "Name: Outcome, Length: 768, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "X_train = np.array(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "y_train = np.array(y_train)\n",
    "X_test = np.array(X_test)\n",
    "y_test = np.array(y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn import svm\n",
    "machine1 = svm.SVC(kernel = 'linear')\n",
    "machine1.fit(X_train,y_train)\n",
    "y_pred = machine1.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "plot_decision_regions(X_train, y_train, machine1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x110b718f208>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.80519480519480524"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "accuracy_score(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n/wN6xKMYfMa3cQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x11090d26828>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(np.array(X)[:,0],np.array(X)[:,1])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "machine2 = svm.SVC(kernel = 'rbf')\n",
    "machine2.fit(X_train,y_train)\n",
    "y_pred2 = machine2.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "plot_decision_regions(np.array(X), np.array(y), machine2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\nYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x110804d4b70>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.68831168831168832"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_test, y_pred2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.\n",
      "  \"This module will be removed in 0.20.\", DeprecationWarning)\n",
      "C:\\ProgramData\\Anaconda3\\lib\\site-packages\\sklearn\\grid_search.py:42: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.\n",
      "  DeprecationWarning)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.grid_search import GridSearchCV\n",
    "from sklearn.svm import SVC\n",
    "pipe_svc = Pipeline([('scl', StandardScaler()),('clf', SVC(random_state=1))])\n",
    "param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]\n",
    "param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},{'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]\n",
    "gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=-1)\n",
    "gs = gs.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.750465549348231\n",
      "{'clf__C': 1.0, 'clf__kernel': 'linear'}\n"
     ]
    }
   ],
   "source": [
    "print(gs.best_score_)\n",
    "print(gs.best_params_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test accuracy: 0.805\n"
     ]
    }
   ],
   "source": [
    ">>> clf = gs.best_estimator_\n",
    ">>> clf.fit(X_train, y_train)\n",
    ">>> print('Test accuracy: %.3f' % clf.score(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.grid_search import GridSearchCV\n",
    "from sklearn.svm import SVC\n",
    "pipe_svc = Pipeline([('scl', StandardScaler()),('clf', SVC(random_state=1))])\n",
    "param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]\n",
    "param_grid = [{'clf__C': param_range,'clf__kernel': ['linear']},{'clf__C': param_range,'clf__gamma': param_range,'clf__kernel': ['rbf']}]\n",
    "gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=-1)\n",
    "gs = gs.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.750465549348231\n",
      "{'clf__C': 1.0, 'clf__kernel': 'linear'}\n"
     ]
    }
   ],
   "source": [
    "print(gs.best_score_)\n",
    "print(gs.best_params_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test accuracy: 0.805\n"
     ]
    }
   ],
   "source": [
    ">>> clf = gs.best_estimator_\n",
    ">>> clf.fit(X_train, y_train)\n",
    ">>> print('Test accuracy: %.3f' % clf.score(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
