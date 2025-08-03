$(document).ready(function () {
    $.get('./media/moviedata/Chinese_Ethnic_Song_and_Dance_Gala/seatmap.json', function (data) {
        console.log(data);
        let drink_data=JSON.parse(window.JSON.stringify(data));
        console.log(data["rows"]);
        //alert(data["rows"][0]["seats"][0].number);
        $(data["rows"]).each(function () {
            console.log(this.rowLabel);
        });
        var innerhtml = `
                <svg viewBox="-15 -15 315 180" class="seatplan seatplan">
                    <g>
                        <a>
                            <rect data-v-09ae1cf2="" x="-15" y="-15" rx="2" ry="2" width="315" height="10" style="fill: rgb(0, 0, 0); "></rect> 
                            <text data-v-09ae1cf2="" x="135" y="-10" dominant-baseline="central" text-anchor="middle" style="font-size: 7px; fill: rgb(255,255,255);">
                            銀幕</text>
                        </a>
                    </g> 
            `;
            var y=-15,rownum=-10.5;
        $(data["rows"]).each(function () {
            y+=15;
            rownum+=15;
            var x=0;
            console.log(this.rowLabel);
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
                let status=this.status;
                if(status=="available") status="#66ff00";
                else status="#ff3300";
                x+=15;
                //alert(this.number+this.status);
                 innerhtml = innerhtml + `
                    <a>
                        <rect  onclick="select_seat(this.id)" id="${this.id}" x="${x}" y="${y}" rx="2" ry="2" width="10" height="10" transform="" style="fill: ${status}; stroke-width: 1; stroke:  ${status}; cursor: pointer;"></rect> <!----> 
                        <text  x="${x+5}" y="${y+4.5}" transform="" dominant-baseline="central" text-anchor="middle" style="font-size: 5px; fill: rgb(0, 0, 0);">
                            ${this.number}
                        </text>
                    </a>`;
            });
            innerhtml = innerhtml +`
                </g>
            `;
        });
        $('#seatmap').append(innerhtml);

        innerhtml=`
            <svg viewBox="-100 0 315 50" class="seatplan seatplan">
                <g>
                    <rect  x="0" y="5" rx="2" ry="2" width="10" height="10" transform="" style="fill:#66ff00; stroke-width: 1; stroke:#66ff00; cursor: pointer;"></rect>
                    <text  x="25" y="9.5" transform="" dominant-baseline="central" text-anchor="middle" style="font-size: 5px; fill: rgb(0, 0, 0);">
                        Available
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

    $.get('./media/jsondata/movielist.json', function (data) {
        var id="polyu01",filmdata;
        $(data).each(function () {
            if(this.id==id){
                filmdata=this;
                return;
            }
        });
            let innerhtml=` <img src="${filmdata.image}" class="w-75 h-75 border rounded" style="object-fit:cover;aspect-ratio: 9 / 14;" alt="${filmdata.name}" style="height:100%">`;
            $("#image").append(innerhtml);
            innerhtml = `
                <h3 class="card-title fw-bold">${filmdata.name}</h3>
                <h4 class="badge text-bg-primary">${filmdata.type}</h4>
                <p class="card-text"><span class="fw-bold">price: </span>${filmdata.price}up</p>
                <p class="card-text"><span class="fw-bold">Showing data: </span>${filmdata.data}</p>
                <p class="card-text"><span class="fw-bold">Venue: </span> ${filmdata.venue}</p>
                <p style=""><span class="fw-bold">Introduction: </span>${filmdata.introduction}</p>`;
            $('#film_info').append(innerhtml);
    }).fail(function (error) {
        let innerhtml = `
         <div class="col mb-3">
            <div class="alert alert-danger" role="alert">
                Failed to fetch drink menu. Please try again later.
            </div>
        </div>`;
        $('#menu_container').append(innerhtml);
    });
});
function select_seat(id){
    
    alert(id);
}
