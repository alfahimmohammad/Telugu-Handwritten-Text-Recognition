import 'package:flutter/material.dart';
import 'dart:io';
import 'package:async/async.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart';
import 'package:toast/toast.dart';

class Displayimage extends StatefulWidget {

  @override
  _DisplayimageState createState() => _DisplayimageState();
}

class _DisplayimageState extends State<Displayimage> {

  uploadImageToServer(File imageFile, BuildContext context) async //_controller,
  {
      //var ipAddress = _controller.text;
      print("attempting to connect to server……");
      var stream = new http.ByteStream(DelegatingStream.typed(imageFile.openRead()));
      var length = await imageFile.length();
      print(length);
      var uri = Uri.parse('http://192.168.1.6:5000/upload');//https://telugu-htr.herokuapp.com/upload, http://192.168.1.6:5000/upload, http://ec2-13-58-170-194.us-east-2.compute.amazonaws.com:8080/upload
      print("connection established");
      Toast.show('wait for a few seconds', context, gravity: Toast.BOTTOM, duration: Toast.LENGTH_LONG);
      var request = new http.MultipartRequest("POST", uri);
      var multipartFile = new http.MultipartFile('file', stream, length,
        filename: basename(imageFile.path));
      request.files.add(multipartFile);
      var response = await request.send();
      print(response.statusCode);
      Toast.show('done', context, gravity: Toast.BOTTOM, duration: Toast.LENGTH_LONG);
      final st = await response.stream.bytesToString();
      print(st);
      Navigator.pushNamed(context, '/result',arguments: {
        'last': st
      });
      return 'hi';
  }

  Widget getWidget(File image, BuildContext  contex){ // _controller,
    if(image == null){
      return Text('No Image Selected',
      style: TextStyle(fontSize: 20.0));
    }
    else{
     return Column(
       children: <Widget>[
         Image.file(image,height: 400.0,width: 350.0),
         /*TextField(controller: _controller,
          decoration: InputDecoration(
            enabledBorder: OutlineInputBorder(
              borderSide: BorderSide(color: Colors.red),
            ),
            hintText: " server's IP address"
          ),
        ),*/
         FloatingActionButton(onPressed: () =>{uploadImageToServer(image, contex)}, //_controller,
         child: Icon(Icons.arrow_upward),)
       ],
     );

    }
  }

  Map data = {};
  @override
  Widget build(BuildContext context) {
    data = ModalRoute.of(context).settings.arguments;
    File image = data['img'];
    //final myController = TextEditingController();
    return Scaffold(
      appBar: AppBar(
        title: Text('Display Image'), 
      ),
      body: ListView(
        children: <Widget>[
          getWidget(image, context),//myController,
        ] ,
      ),
    );
  }
}