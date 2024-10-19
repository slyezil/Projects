function myFunction(){    
    document.getElementById("demo").innerHTML = "sing";
}
function second(){

    var x;
    console.log(x);
    x = inside();
    document.getElementById("but").innerHTML = x;
    
    }
function inside (){
    let x = 3;
    console.log(x);
    return  x ;

}

 const button    =   document.getElementById("button1");
 button.addEventListener('click',function() { alert("do not click");});

