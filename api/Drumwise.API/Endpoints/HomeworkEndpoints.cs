using Drumwise.Application.Common.Extensions;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models.Identity;
using Drumwise.Features.Homeworks;
using MediatR;
using Microsoft.AspNetCore.Mvc;

namespace Drumwise.API.Endpoints;

public static class HomeworkEndpoints
{
    public static void MapHomeworkEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var homeworkGroup = endpoints.MapGroup("/homework").RequireAuthorization();
        
        homeworkGroup.MapPost("/",  async Task<IResult>
            ([FromBody] CreateHomeworkCommand createHomeworkCommand, [FromServices]ISender sender) =>
            {
                var (result, homeworkId) = await sender.Send(createHomeworkCommand).ConfigureAwait(false);

                return result.ProduceApiResponse(homeworkId);
            })
            .RequireAuthorization("CanAddHomework")
            .Produces<Guid>();

        homeworkGroup.MapGet("/", async Task<IResult>
            ([AsParameters] GetHomeworksWithPaginationQuery query, [FromServices] ISender sender) =>
        {
            var (result, homeworks) = await sender.Send(query).ConfigureAwait(false);

            return result.ProduceApiResponse(homeworks);
        })
        .Produces<HomeworkItemBriefDto>();
    }
}