import 'package:flutter/material.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'package:toast/toast.dart';
import 'dart:async';


class Result extends StatefulWidget {
  @override
  _ResultState createState() => _ResultState();
}

class _ResultState extends State<Result> {
  Map dat = {};
  Future <void> _write(String text, context) async {
  final directory = await getExternalStorageDirectory();
  final path = join(directory.path,'telugu_text.txt');
  final file = File(path);
  await file.writeAsString(text);
  Toast.show('stored in $path',context,duration: Toast.LENGTH_LONG);
  }
  @override
  Widget build(BuildContext context) {
    dat = ModalRoute.of(context).settings.arguments;
    return Scaffold(
      appBar: AppBar(
        title: Text('Result'),
      ),
      body: ListView(
        children: <Widget>[
          SelectableText(dat['last'] != null ? dat['last'] : "No Content found",//Text(dat['last'] != null ? dat['last'] : "No Content found",
            style: TextStyle(fontSize: 20.0)),
          RaisedButton(onPressed: () => {
            if (dat['last'] != null){
              _write(dat['last'],context)
            }
            else{
              Toast.show('No content to save', context, gravity: Toast.BOTTOM)
            }
          },
            child: Text('Save as a .txt file'),)
        ]
      ),
    );
  }
}