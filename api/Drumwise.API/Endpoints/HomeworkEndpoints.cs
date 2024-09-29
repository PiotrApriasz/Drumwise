using Drumwise.Application.Common.Extensions;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models.Identity;
using Drumwise.Features.Homeworks;
using Drumwise.Infrastructure.Identity.Constants;
using MediatR;
using Microsoft.AspNetCore.Mvc;

namespace Drumwise.API.Endpoints;

public static class HomeworkEndpoints
{
    public static void MapHomeworkEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var homeworkGroup = endpoints.MapGroup(ApiPaths.HomeworkRootApi).RequireAuthorization();
        
        homeworkGroup.MapPost(ApiPaths.CreateHomework,  async Task<IResult>
            ([FromBody] CreateHomeworkCommand createHomeworkCommand, [FromServices]ISender sender) =>
            {
                var (result, homeworkId) = await sender.Send(createHomeworkCommand).ConfigureAwait(false);

                return result.ProduceApiResponse(homeworkId);
            })
            .RequireAuthorization(Policies.CanAddHomework)
            .Produces<Guid>();

        homeworkGroup.MapGet(ApiPaths.GetAllHomeworks, async Task<IResult>
            ([AsParameters] GetHomeworksWithPaginationQuery query, [FromServices] ISender sender) =>
        {
            var (result, homeworks) = await sender.Send(query).ConfigureAwait(false);

            return result.ProduceApiResponse(homeworks);
        })
        .Produces<HomeworkItemBriefDto>();

        homeworkGroup.MapGet(ApiPaths.GetHomeworkById, async Task<IResult>
            ([AsParameters] GetHomeworkQuery query, [FromServices] ISender sender) =>
        {
            var (result, homework) = await sender.Send(query);

            return result.ProduceApiResponse(homework);
        })
        .Produces<HomeworkItemDto>()
        .ProducesValidationProblem(StatusCodes.Status404NotFound);
    }
}