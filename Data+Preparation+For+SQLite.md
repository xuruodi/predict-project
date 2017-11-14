
## 1. Data Introduction


```python
import pandas as pd
academy_awards=pd.read_csv("academy_awards.csv",encoding="ISO-8859-1")
academy_awards.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Category</th>
      <th>Nominee</th>
      <th>Additional Info</th>
      <th>Won?</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010 (83rd)</td>
      <td>Actor -- Leading Role</td>
      <td>Javier Bardem</td>
      <td>Biutiful {'Uxbal'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010 (83rd)</td>
      <td>Actor -- Leading Role</td>
      <td>Jeff Bridges</td>
      <td>True Grit {'Rooster Cogburn'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010 (83rd)</td>
      <td>Actor -- Leading Role</td>
      <td>Jesse Eisenberg</td>
      <td>The Social Network {'Mark Zuckerberg'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010 (83rd)</td>
      <td>Actor -- Leading Role</td>
      <td>Colin Firth</td>
      <td>The King's Speech {'King George VI'}</td>
      <td>YES</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010 (83rd)</td>
      <td>Actor -- Leading Role</td>
      <td>James Franco</td>
      <td>127 Hours {'Aron Ralston'}</td>
      <td>NO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(academy_awards.shape)
print(academy_awards.dtypes)
```

    (10137, 11)
    Year               object
    Category           object
    Nominee            object
    Additional Info    object
    Won?               object
    Unnamed: 5         object
    Unnamed: 6         object
    Unnamed: 7         object
    Unnamed: 8         object
    Unnamed: 9         object
    Unnamed: 10        object
    dtype: object
    


```python
academy_awards.columns
```




    Index(['Year', 'Category', 'Nominee', 'Additional Info', 'Won?', 'Unnamed: 5',
           'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10'],
          dtype='object')




```python
print(academy_awards['Unnamed: 5'].value_counts())
print(academy_awards['Unnamed: 6'].value_counts())
print(academy_awards['Unnamed: 7'].value_counts())
print(academy_awards['Unnamed: 8'].value_counts())
print(academy_awards['Unnamed: 9'].value_counts())
print(academy_awards['Unnamed: 10'].value_counts())
```

    *                                                                                                               7
     resilience                                                                                                     1
     discoverer of stars                                                                                            1
     D.B. "Don" Keele and Mark E. Engebretson has resulted in the over 20-year dominance of constant-directivity    1
     error-prone measurements on sets. [Digital Imaging Technology]"                                                1
    Name: Unnamed: 5, dtype: int64
    *                                                                   9
     direct radiator bass style cinema loudspeaker systems. [Sound]"    1
     sympathetic                                                        1
     flexibility and water resistance                                   1
    Name: Unnamed: 6, dtype: int64
    *                                                     1
     while requiring no dangerous solvents. [Systems]"    1
     kindly                                               1
    Name: Unnamed: 7, dtype: int64
    *                                                 1
     understanding comedy genius - Mack Sennett.""    1
    Name: Unnamed: 8, dtype: int64
    *    1
    Name: Unnamed: 9, dtype: int64
    *    1
    Name: Unnamed: 10, dtype: int64
    

##### As there is only a little useful information in these unname columns,I prepare to drop them later.
在这些未命名的列里只有很少有用的信息，我打算后期删掉它们。

## 2. Data Cleaning And Filtering


```python
academy_awards["Year"]=academy_awards["Year"].str[0:4]
academy_awards["Year"]=academy_awards["Year"].astype("int64")
later_than_2000=academy_awards[academy_awards["Year"]>2000]
```


```python
later_than_2000["Category"]
```




    0                                   Actor -- Leading Role
    1                                   Actor -- Leading Role
    2                                   Actor -- Leading Role
    3                                   Actor -- Leading Role
    4                                   Actor -- Leading Role
    5                                Actor -- Supporting Role
    6                                Actor -- Supporting Role
    7                                Actor -- Supporting Role
    8                                Actor -- Supporting Role
    9                                Actor -- Supporting Role
    10                                Actress -- Leading Role
    11                                Actress -- Leading Role
    12                                Actress -- Leading Role
    13                                Actress -- Leading Role
    14                                Actress -- Leading Role
    15                             Actress -- Supporting Role
    16                             Actress -- Supporting Role
    17                             Actress -- Supporting Role
    18                             Actress -- Supporting Role
    19                             Actress -- Supporting Role
    20                                  Animated Feature Film
    21                                  Animated Feature Film
    22                                  Animated Feature Film
    23                                          Art Direction
    24                                          Art Direction
    25                                          Art Direction
    26                                          Art Direction
    27                                          Art Direction
    28                                         Cinematography
    29                                         Cinematography
                                  ...                        
    1264                                              Writing
    1265                                              Writing
    1266                                       Honorary Award
    1267                                       Honorary Award
    1268                     Jean Hersholt Humanitarian Award
    1269    Scientific and Technical (Scientific and Engin...
    1270    Scientific and Technical (Scientific and Engin...
    1271    Scientific and Technical (Scientific and Engin...
    1272    Scientific and Technical (Scientific and Engin...
    1273    Scientific and Technical (Scientific and Engin...
    1274    Scientific and Technical (Scientific and Engin...
    1275    Scientific and Technical (Scientific and Engin...
    1276    Scientific and Technical (Technical Achievemen...
    1277    Scientific and Technical (Technical Achievemen...
    1278    Scientific and Technical (Technical Achievemen...
    1279    Scientific and Technical (Technical Achievemen...
    1280    Scientific and Technical (Technical Achievemen...
    1281    Scientific and Technical (Technical Achievemen...
    1282    Scientific and Technical (Technical Achievemen...
    1283    Scientific and Technical (Technical Achievemen...
    1284    Scientific and Technical (Technical Achievemen...
    1285    Scientific and Technical (Technical Achievemen...
    1286    Scientific and Technical (Technical Achievemen...
    1287    Scientific and Technical (Technical Achievemen...
    1288    Scientific and Technical (Technical Achievemen...
    1289    Scientific and Technical (Technical Achievemen...
    1290    Scientific and Technical (Gordon E. Sawyer Award)
    1291              Scientific and Technical (Bonner Medal)
    1292            Scientific and Technical (Special Awards)
    1293            Scientific and Technical (Special Awards)
    Name: Category, dtype: object




```python
award_categories=["Actor -- Leading Role","Actor -- Supporting Role",
                 "Actress -- Leading Role","Actress -- Supporting Role"]
nominations=later_than_2000[later_than_2000["Category"].isin(award_categories)]
won_dict={"YES":1,"NO":0}
nominations['Won?']=nominations['Won?'].map(won_dict)
nominations["Won"]=nominations["Won?"]
```

    C:\Users\xuruoqian\Anaconda3\lib\site-packages\ipykernel\__main__.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    C:\Users\xuruoqian\Anaconda3\lib\site-packages\ipykernel\__main__.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    


```python
nominations.columns
```




    Index(['Year', 'Category', 'Nominee', 'Additional Info', 'Won?', 'Unnamed: 5',
           'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10',
           'Won'],
          dtype='object')




```python
final_nominations=nominations.drop(['Won?', 'Unnamed: 5',
       'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10'],axis=1)
```


```python
info_one=final_nominations["Additional Info"].str.rstrip("'}")
info_two=info_one.str.split(" {'")
final_nominations["movie"]=info_two.str[0]
final_nominations["character"]=info_two.str[1]
final_nominations=final_nominations.drop('Additional Info',axis=1)
```

## 3. Exporting To SQLite


```python
import sqlite3
conn=sqlite3.connect("nominations.db")
conn.execute('drop table nomination5;')
final_nominations.to_sql("nomination5",conn,index=False)
```


```python
first_ten=conn.execute('select * from nomination5 limit 10;')
schema=conn.execute('pragma table_info(nomination5);').fetchall()
```


```python
for row in first_ten:
    print(row)
```

    (2010, 'Actor -- Leading Role', 'Javier Bardem', 0, 'Biutiful', 'Uxbal')
    (2010, 'Actor -- Leading Role', 'Jeff Bridges', 0, 'True Grit', 'Rooster Cogburn')
    (2010, 'Actor -- Leading Role', 'Jesse Eisenberg', 0, 'The Social Network', 'Mark Zuckerberg')
    (2010, 'Actor -- Leading Role', 'Colin Firth', 1, "The King's Speech", 'King George VI')
    (2010, 'Actor -- Leading Role', 'James Franco', 0, '127 Hours', 'Aron Ralston')
    (2010, 'Actor -- Supporting Role', 'Christian Bale', 1, 'The Fighter', 'Dicky Eklund')
    (2010, 'Actor -- Supporting Role', 'John Hawkes', 0, "Winter's Bone", 'Teardrop')
    (2010, 'Actor -- Supporting Role', 'Jeremy Renner', 0, 'The Town', 'James Coughlin')
    (2010, 'Actor -- Supporting Role', 'Mark Ruffalo', 0, 'The Kids Are All Right', 'Paul')
    (2010, 'Actor -- Supporting Role', 'Geoffrey Rush', 0, "The King's Speech", 'Lionel Logue')
    


```python
for row in schema:
    print(row)
```

    (0, 'Year', 'INTEGER', 0, None, 0)
    (1, 'Category', 'TEXT', 0, None, 0)
    (2, 'Nominee', 'TEXT', 0, None, 0)
    (3, 'Won', 'INTEGER', 0, None, 0)
    (4, 'movie', 'TEXT', 0, None, 0)
    (5, 'character', 'TEXT', 0, None, 0)
    


```python
conn.execute('drop table ceremonies;')
conn.execute('create table ceremonies(id integer primary key,Year integer,Host text);')
```




    <sqlite3.Cursor at 0x2191aa52500>




```python
years_hosts = [(2010, "Steve Martin"),
               (2009, "Hugh Jackman"),
               (2008, "Jon Stewart"),
               (2007, "Ellen DeGeneres"),
               (2006, "Jon Stewart"),
               (2005, "Chris Rock"),
               (2004, "Billy Crystal"),
               (2003, "Steve Martin"),
               (2002, "Whoopi Goldberg"),
               (2001, "Steve Martin"),
               (2000, "Billy Crystal"),
            ]
insert_query='insert into ceremonies (Year,Host) values(?,?);'
conn.executemany(insert_query,years_hosts)
```




    <sqlite3.Cursor at 0x2191aa52650>




```python
conn.execute('select * from ceremonies limit 10;').fetchall()
```




    [(1, 2010, 'Steve Martin'),
     (2, 2009, 'Hugh Jackman'),
     (3, 2008, 'Jon Stewart'),
     (4, 2007, 'Ellen DeGeneres'),
     (5, 2006, 'Jon Stewart'),
     (6, 2005, 'Chris Rock'),
     (7, 2004, 'Billy Crystal'),
     (8, 2003, 'Steve Martin'),
     (9, 2002, 'Whoopi Goldberg'),
     (10, 2001, 'Steve Martin')]




```python
conn.execute('pragma table_info(ceremonies);').fetchall()
```




    [(0, 'id', 'integer', 0, None, 1),
     (1, 'Year', 'integer', 0, None, 0),
     (2, 'Host', 'text', 0, None, 0)]




```python
conn.execute('pragma foreign_keys=on').fetchall()
```




    []




```python
conn.execute('''create table nominations_four(id integer primary key,
                                              category text,
                                              nominee text,
                                              movie text,
                                              character text,
                                              won text,
                                              ceremony_id integer,
                                              foreign key(ceremony_id) references ceremonies(id));''').fetchall()
```




    []




```python
query='''select ceremonies.id as ceremony_id,
                nomination5.category as category,
                nomination5.nominee as nominee,
                nomination5.movie as movie,
                nomination5.character as character,
                nomination5.won as won 
         from nomination5 
         inner join ceremonies 
         on nomination5.year==ceremonies.year;'''
join_nominations=conn.execute(query).fetchall()
```


```python
insert_query2='insert into nominations_four(ceremony_id,category,nominee,movie,character,won) values(?,?,?,?,?,?);'
conn.executemany(insert_query2,join_nominations)
```




    <sqlite3.Cursor at 0x2191aa52ab0>




```python
conn.execute('select * from nominations_four limit 5').fetchall()
```




    [(1, 'Actor -- Leading Role', 'Javier Bardem', 'Biutiful', 'Uxbal', '0', 1),
     (2,
      'Actor -- Leading Role',
      'Jeff Bridges',
      'True Grit',
      'Rooster Cogburn',
      '0',
      1),
     (3,
      'Actor -- Leading Role',
      'Jesse Eisenberg',
      'The Social Network',
      'Mark Zuckerberg',
      '0',
      1),
     (4,
      'Actor -- Leading Role',
      'Colin Firth',
      "The King's Speech",
      'King George VI',
      '1',
      1),
     (5,
      'Actor -- Leading Role',
      'James Franco',
      '127 Hours',
      'Aron Ralston',
      '0',
      1)]




```python
conn.execute('drop table nominations;')
```




    <sqlite3.Cursor at 0x2191aa526c0>




```python
conn.execute('alter table nominations_four rename to nominations')
```




    <sqlite3.Cursor at 0x2191aa528f0>




```python
conn.execute('drop table movies;')
conn.execute('drop table actors;')
conn.execute('create table movies(id integer primary key,movie text);')
conn.execute('create table actors(id integer primary key,actor text);')
```




    <sqlite3.Cursor at 0x2191aa52e30>




```python
conn.execute('drop table movie_actor;')
conn.execute('''create table movie_actor(id integer primary key,
                                         movie_id integer references movies(id),
                                         actor_id integer references actors(id));''')
```




    <sqlite3.Cursor at 0x2191aa52f10>




```python
conn.execute('select * from movie_actor;')
```




    <sqlite3.Cursor at 0x2191aa52ea0>


