import 'package:flutter/material.dart';
import 'pages/home.dart';
import 'pages/cropimage.dart';
import 'pages/displayimage.dart';
import 'pages/result.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      initialRoute: '/home',
      routes: {
        '/home': (context) => Home(),
        '/cropimage': (context) => Cropimage(),
        '/displayimage': (context) => Displayimage(),
        '/result': (context) => Result(),
      }
    );
  }
}
