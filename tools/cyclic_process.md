**[TOOL cyclic_process]**

**name:** cyclic_process

**code:** 
```code
/work#cyclic_process.sh
```

**example**:
```code			 
/work#cyclic_process.sh myfile.dat -n 5 
```

**description:** циклически обработать части файла

параметры
* **FILE** имя файла, из которого образованы части (пример: FILE=myfile.dat для myfile_1.dat myfile_2.dat ...)
* **-n COUNT** количество файлов для чтения

Результаты в <имя FILE без расширения>_out.txt

**manager:** true

**[/TOOL]**

