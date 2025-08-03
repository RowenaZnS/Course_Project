var seat_id_list=[];
var tickets=[];
var name2;
var movie_imagel;
var id;
// <!-- XU Zhuoning liu xinyan 21101256d;22097739d -->
$(document).ready(function () {
    var color;
    
    $.ajax({
        url: '/event/movie',
        type: 'GET',
        success: function (data) {
            if (data.status === 'success') {
                id = data.movies.id;
                $.get('./media/jsondata/movielist.json', function (data) {
                    var filmdata;
                    $(data).each(function () {
                        if (this.id == id) {
                            filmdata = this;
                            return;
                        }
                    });
                    name2=filmdata.name;
                    movie_imagel=filmdata.image;
                    color = filmdata.seat_type;
                    let innerhtml = ` <img src="${filmdata.image}" class="w-75 h-75 border rounded" style="object-fit:cover;aspect-ratio: 9 / 14;" alt="${filmdata.name}" style="height:100%">`;
                    $("#image").append(innerhtml);
                    var date = "";
                    $(filmdata.date).each(function () {
                        date = date + this.general_date + " ";
                    });
                    innerhtml = `
                       <h3 class="card-title fw-bold my-4">${filmdata.name}</h3>
                       <h4 class="badge text-bg-primary mb-4">${filmdata.type}</h4>               
                       <h4 class="card-text mb-4"><span class="fw-bold">Showing data: </span>${date}</h4>
                       <h4 class="card-text mb-4"><span class="fw-bold">Venue: </span> ${filmdata.venue}</h4>`;
                    $('#film_info').append(innerhtml);
                    innerhtml = `<p class="card-text"><span class="fw-bold">price: </span>${filmdata.price}up</p> 
                   <p style=""><span class="fw-bold">Introduction: </span>${filmdata.introduction}</p>`;
                    $('#film_info').append(innerhtml);
                    $(filmdata.date).each(function () {
                        innerhtml = `
                       <input type="radio" class="btn-check wide-btn form-control"  name="vbtn-radio" id="${this.id}"
                           autocomplete="off" aria-pressed="true" >
                       <label class="btn btn-bd-primary d-block mb-3 wide-btn" for="${this.id}" aria-pressed="true">
                           <div class="event-card">
                               <div class="d-flex mx-2">
                                   <div class="event-date mr-3">
                                       <div class="event-month">${this.month}, ${this.year}</div>
                                       <div class="event-day">${this.day}</div>
                                   </div>
                                   <div class="mx-3" style="text-align: left;">
                                       <div class="text-primary event-time" style="font-size: 20px;">${this.week}, ${this.general_time}</div>
                                       <div style="font-size: 15px;">${filmdata.venue}</div>
                                   </div>
                               </div>
                           </div>
                       </label>
                     `;
                        $("#select_date").append(innerhtml);
                    });
                    $(filmdata.seat_type).each(function () {
                        var innerhtml2 = `
                       <div class="alert " role="alert" style="--bs-alert-bg: ${this.color}; width:100%">
                          <div class="text-center d-inline-block" style="width:70%">${this.name} $${this.price}</div> 
                          <div class="text-end d-inline-block"><span id='${this.color.substring(1)}' name="${this.price}">0</span> Tickets</div>
                       </div>
                    `;
                        let temp={
                            "name":this.name,
                            "price":this.price,
                            "id":this.color.substring(1),
                            "num":0
                        };
                        tickets.push(temp);
                        $("#select_info").append(innerhtml2);
                    });
                    $.get('./media/moviedata/Chinese_Ethnic_Song_and_Dance_Gala/seatmap.json', function (data) {
                        var innerhtml = `
                        <svg viewBox="-15 -15 315 180" class="seatplan seatplan">
                            <g>
                                <a>
                                    <rect data-v-09ae1cf2="" x="-15" y="-15" rx="2" ry="2" width="300" height="10" style="fill: rgb(0, 0, 0); "></rect> 
                                    <text data-v-09ae1cf2="" x="135" y="-10" dominant-baseline="central" text-anchor="middle" style="font-size: 7px; fill: rgb(255,255,255);">
                                    銀幕</text>
                                </a>
                            </g> 
                    `;
                        var y = -15, rownum = -10.5;
                        $(data["rows"]).each(function () {
                            y += 15;
                            rownum += 15;
                            var x = 0;
                            innerhtml += `
                        <g data-v-09ae1cf2="">
                            <text data-v-09ae1cf2="" x="-7.5" y="${rownum}" width="15" height="15" text-anchor="middle" dominant-baseline="central" style="font-size: 5px; fill: rgb(0,0,0); ">
                                ${this.rowLabel}
                            </text> 
                            <text data-v-09ae1cf2="" x="262.5" y="${rownum}" width="15" height="15" text-anchor="middle" dominant-baseline="central" style="font-size: 5px; fill: rgb(0,0,0);">
                                ${this.rowLabel}
                            </text>
                    `;
                            $(this.seats).each(function () {
                                let status = this.status;
                                var seatid = this.id;
                                if (status == "available") {
                                    status = "#66ff00";
                                    $(color).each(function () {
                                        let start = this.start_line;
                                        let end = this.end_line;
                                        if (isCharBetween(seatid.charAt(0), start, end))
                                            status = this.color;
                                    });
                                }
                                else status = "#ff3300";
                                let cursor=`cursor: pointer;`;
                                if(status == "#ff3300"){
                                    cursor=`cursor: not-allowed;`
                                }
                                x += 15;
                                //alert(this.number+this.status);
                                innerhtml = innerhtml + `
                            <a>
                                <rect  onclick="select_seat(this.id)" id="${this.id}" x="${x}" y="${y}" rx="2" ry="2" width="10" height="10" transform="" style="${cursor} fill: ${status}; stroke-width: 1; stroke:  ${status}; "></rect> <!----> 
                                <text  x="${x + 5}" y="${y + 4.5}" transform="" dominant-baseline="central" text-anchor="middle" style="font-size: 5px; fill: rgb(0, 0, 0);">
                                    ${this.number}
                                </text>
                            </a>`;
                            });
                            innerhtml = innerhtml + `
                        </g>
                    `;
                        });
                        $('#seatmap').append(innerhtml);

                        innerhtml = `
                    <svg viewBox="-100 0 315 50" class="seatplan seatplan">
                        <g>
                            <rect  x="0" y="5" rx="2" ry="2" width="10" height="10" transform="" style="fill:#66ff00; stroke-width: 1; stroke:#0d6efd; cursor: pointer;"></rect>
                            <text  x="25" y="9.5" transform="" dominant-baseline="central" text-anchor="middle" style="font-size: 5px; fill: rgb(0, 0, 0);">
                                Selected
                            </text>
                            <rect  x="50" y="5" rx="2" ry="2" width="10" height="10" transform="" style="fill:#ff3300; stroke-width: 1; stroke:#ff3300; cursor: pointer;"></rect>
                            <text  x="85" y="9.5" transform="" dominant-baseline="central" text-anchor="middle" style="font-size: 5px; fill: rgb(0, 0, 0);">
                                Not Available
                            </text>
                        </g>
                    </svg>
                `;
                        $('#seatmap').append(innerhtml);
                    }).fail(function (error) {
                        let innerhtml = `
                 <div class="col mb-3">
                    <div class="alert alert-danger" role="alert">
                        Failed to fetch drink menu. Please try again later.
                    </div>
                </div>`;
                        $('#menu_container').append(innerhtml);
                    });
                }).fail(function (error) {
                    let innerhtml = `
                <div class="col mb-3">
                   <div class="alert alert-danger" role="alert">
                       Failed to fetch drink menu. Please try again later.
                   </div>
               </div>`;
                    $('#menu_container').append(innerhtml);
                });
            } else {
                alert('Please login');
                window.open('/login.html', '_self');
            }
        },
        error: function () {
            alert('Please login');
            window.open('/login.html', '_self');
        },
    });
    $('#confirm').on('click', function(){
        console.log(tickets);
        var formData = new FormData();
        var radio = document.getElementsByName("vbtn-radio");
        var time_select;
	    for (let i=0; i<radio.length; i++) {
            if (radio[i].checked) {
                time_select=$(radio[i]).attr('id');
            }
        }
        formData.append('Event_id', `${id}`);
        formData.append('movie_imagel', `${movie_imagel}`);
        formData.append('movie_name', `${name2}`);
        formData.append('time_select', `${time_select}`);
        formData.append('seat_list', seat_id_list);
        formData.append('tickets', JSON.stringify(tickets));
        formData.append('total_cost',  $('#total_cost').val());
        $.ajax({
            url: '/pay/payment_page',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (data) {
              if (data.status === 'success') {
                alert(`Your payment request is accepted`);
                window.location.href = '/payment.html';
              } else {
                alert(data.message);
              }
            },
            error: function (xhr) {
              const response = xhr.responseJSON || JSON.parse(xhr.responseText);
              if (response && response.message) {
                alert(response.message);
              } else {
                alert('Unknown error');
              }
              console.error('Error:', response);
            },
          });
    });



});
function select_seat(id) {
    if(rgb2hex($(`#${id}`).css("stroke"))!='#0d6efd'& rgb2hex($(`#${id}`).css("fill"))!='#ff3300'){
        var result = confirm(`Are you sure to book seat ${id}`);  
        if(result){  
            alert(`Succesfully booked seat ${id}`);  
        }else{  
              return;
        }  
        var tickets_id=rgb2hex($(`#${id}`).css("fill"));
        var num=Number($(`${tickets_id}`).html());
        $(`${tickets_id}`).html(num+1);
        $(`#${id}`).css("stroke",'#0d6efd');
        $(`#${id}`).css("stroke-width",1);
        seat_id_list.push(id);
        $(tickets).each(function(){
            if(`#${this.id}`==tickets_id){
                this.num+=1;
                return;
            }
        });
    }
    else if(rgb2hex($(`#${id}`).css("fill"))=='#ff3300'){
        alert(`Seat ${id} is unavailable`); 
        return; 
    }
    else{
        var result2 = confirm(`Are you sure to cancel booking seat ${id}`);  
        if(result2){  
            alert(`Succesfully cancel seat ${id}`);  
        }else{  
              return;
        } 
        var tickets_id2=rgb2hex($(`#${id}`).css("fill"));
        var num2=Number($(`${tickets_id}`).html());
        var res=num2-1;
        if (isNaN(res)) res = 0;
        $(`${tickets_id2}`).html(res);
        $(`#${id}`).css("stroke",`${tickets_id2}`);
        $(`#${id}`).css("stroke-width",1);     
        seat_id_list = seat_id_list.filter(item => item != id)
        $(tickets).each(function(){

            if(`#${this.id}`==tickets_id2){
                if (isNaN(res)) res = 0;
                this.num=res;
                return;
            }
        });
    }
    $('#total_cost').val(cal_cost());
}
function isCharBetween(char, start, end) {
    // Ensure the input is a single character
    if (char.length !== 1) {
        return false;
    }
    // Convert the character to uppercase for case-insensitivity
    const upperChar = char.toUpperCase();
    // Check if the character is between 'A' and 'D'
    return upperChar >= start && upperChar <= end;
}
function rgb2hex(rgb){
    rgb = rgb.match(/^rgba?[\s+]?\([\s+]?(\d+)[\s+]?,[\s+]?(\d+)[\s+]?,[\s+]?(\d+)[\s+]?/i);
    return (rgb && rgb.length === 4) ? "#" +
     ("0" + parseInt(rgb[1],10).toString(16)).slice(-2) +
     ("0" + parseInt(rgb[2],10).toString(16)).slice(-2) +
     ("0" + parseInt(rgb[3],10).toString(16)).slice(-2) : '';
   }
function cal_cost(){
    let cost=0;
    $(tickets).each(function(){
        let res=this.num;
        if (isNaN(res)) res = 0;
        cost+=Number(this.num)*Number(this.price);       
    });
    return cost;
}