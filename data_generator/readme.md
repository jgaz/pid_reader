## DWG

Files: 2079

Mainly:
- ./STATOIL_STD_DWG/Statoil_STD_dwg/
- ./STANDARD_COMPANY_V8I/Standard_Company_V8i/workspace/Company/

## DGN

Files: 53

Mainly:
- ./STANDARD_COMPANY_V8I/Standard_Company_V8i/workspace/

## CEL

Files: 70

Mainly:
- ./STANDARD_COMPANY_V8I/Standard_Company_V8i/workspace/Company/Cell/



## Conversion path

Collect ccf files in list_of_symbols_ccf

Use [ODAFileConverter](https://www.opendesign.com/guestfiles/oda_file_converter)
to convert from DWG to DXF (2007 ASCII DXF)

Convert to PDF
```bash
find ./ -name '*.dwg' -exec cp {} ./dwg_library/  \;
rm *.pdf && librecad dxf2pdf -a -k *.dxf
```

Convert to PNG
```bash
gs -sDEVICE=png16m -dNOPAUSE -dBATCH -dSAFER -r300 -sOutputFile="./png/$filename.png" "$filename"
```


## Big diagrams conversion

PDF -> PNG

- 150 resolution.
- 5000x3500 px.


## Extract a list of symbols with an explanation

`find ./ -name *.ccf -exec cat {} \;|grep -v '#' >list_of_symbols_ccf.txt`

Folder structure of ./STATOIL_STD_DWG/Statoil_STD_dwg/ gives also a context to the found symbols


## Symbols types

### Symbols that contain text

- ISCD-F00X: 5 text special spacing?
- ISCD-R00x: 1 text
- ISCD-S00x: 1-3 Text
- STJI00x: 1-2 Text
- STJM00x: 1-2 Text



### Symbols that do not contain text
