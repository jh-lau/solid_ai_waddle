1. 闭包是什么现象，有什么应用？
    - 一个能记住嵌套作用域变量值的函数，尽管作用域已经不存在，即：闭包引用了自由变量，即使生成闭包的环境已经释放，闭包依然存在。
    - 闭包的三要素
        - 含有嵌套函数
        - 嵌套（内嵌）函数引用外部变量
        - 嵌套函数被返回
    - 应用
        - 运行时可以有多个实例，即使传入的参数相同
        - 模拟构造类
        - 实现偏函数的功能
2. 可迭代对象和迭代器是什么区别
    - 可迭代对象`Iterable`，即有迭代能力的对象，实现了魔法函数`__iter__`，**通常**该函数返回一个实现了`__next__`方法的对象，当然这个返回不是必须的；迭代器`Iterator`同时实现了`__iter__`和`__next__`。
    - `迭代器是实现了next方法的可迭代对象。调用iter方法可以将可迭代对象变成迭代器`，这个说法不准确，可迭代对象还是可迭代对象，是否为迭代器取决于调用`iter()`方法的返回结果。

100. holder